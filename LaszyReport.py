import os
import re
import glob
import json
import tqdm
import math
import datetime
import pandas as pd

import Laszy
from lidar.lidar_const import RegexLidar


# ------------------------------------
# -- LaszyReport constants
# ------------------------------------
CORRUPT_FILE_MSG = "POSSIBLE CORRUPT FILE (Failed to decompress)"
ACQUISITON = False  # !!! Temporary constant - will be removed in the future. Being used to switch off some features.
LASZY_REPORT_DROP_COLUMNS = [  # drop all columns that don't need to be checked for issues
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "guid_hex", "generating_software", "point_count",
    'waveform_internal_packets', 'waveform_external_packets', 'projection', 'spheroid', "wkt_bbox",
    'vert_cs', 'proj_cs', 'geog_cs', 'vlr_count', 'vlr_has_geotiff_crs', 'date_start',
    'has_keypoint', 'has_withheld', 'has_overlap', 'evlr_count', 'evlr_has_geotiff_crs', "rgb_encoding"
]


# =========================================================
# --- LaszyReport class
# =========================================================
class _LaszyReportColumns:

    """
    'enum' class containing constant lists for LaszyReport columns.
    """

    FILENAME = "filename"

    PUB_HDR = [
        "guid_asc", "guid_hex", "file_source_id", "system_id",  "generating_software", "creation_date",
        "version", "point_data_format", "point_count",  "x_min",  "x_max",  "y_min", "y_max", "z_min",
        "z_max", "x_scale", "y_scale", "z_scale", "x_offset", "y_offset", "z_offset"
    ]
    GLOBAL_ENCODING = [
        'global_encoding', 'gps_standard_time', 'waveform_internal_packets',
        'waveform_external_packets', 'synthetic_returns', 'wkt_crs'
    ]
    CRS = [
        'projection', 'vert_datum', 'compd_cs', 'spheroid', 'hz_datum',
        'vert_cs', 'proj_cs', 'geog_cs'
    ]
    VLR_HDR = [
        'vlr_count', 'vlr_has_wkt_crs', 'vlr_has_geotiff_crs'
    ]
    POINT_RECORDS = [
        'classes', 'gps_time_min', 'gps_time_max', 'date_start', 'date_end',
        'flightline_start', 'flightline_end'
    ]
    CLASS_FLAGS = [
        'has_synthetic', 'has_keypoint', 'has_withheld', 'has_overlap'
    ]
    EVLR_HDR = [
        'evlr_count', 'evlr_has_wkt_crs', 'evlr_has_geotiff_crs'
    ]
    RGB_ENCODING = "rgb_encoding"
    WKT_BBOX = "wkt_bbox"

    COLUMNS = [
        FILENAME, *PUB_HDR, *GLOBAL_ENCODING, *CRS, *VLR_HDR,
        *POINT_RECORDS, *CLASS_FLAGS, *EVLR_HDR, RGB_ENCODING, WKT_BBOX
    ]


class LaszyReport:

    def __init__(self, flist: list[str] = None, odir: str = ".", to_json: bool = False, verbose: bool = False):

        """
        Initialize LaszyReport object.

        The argument flist may contain:
            - LAS/LAZ files
            - JSON files holding a Laszy summary (generated from Laszy.summary())
            - Both LAS/LAZ files and JSON files.

        Note that an 'flist' containing both json and las/laz files will be partitioned
        into self.json_list, and self.lidar_list.

        :param flist: A list containing input files.
        :param odir: Out directory for tabular dataset (default=".")
        :param to_json: When 'True', will write a json summary file for input LAS/LAZ files.
        :param verbose: When 'True', display information about progress to the user.
        """

        self._path = ""
        self._errors = []
        self.outdir = odir
        self.verbose = verbose
        self._json_completed = []
        self._lidar_completed = []
        self.file_list = flist
        self.las_to_json = to_json
        self._DEFAULT_NAME = "laszy_report.csv"
        self._JSON_LOG_NAME = "json_completed.log"
        self._LIDAR_LOG_NAME = "lidar_completed.log"
        self.json_list = [f for f in flist if f.endswith('json')]
        self.lidar_list = [f for f in flist if (f.endswith("laz") or f.endswith("las"))]
        self.__remove_processed_lidar()

    def __remove_processed_lidar(self):
        laszy_json = os.path.join(self.outdir, "laszy_json")
        if os.path.exists(laszy_json):
            self.json_list.extend(
                glob.glob(os.path.join(laszy_json, "*.json"))
            )

        laszy_json_bases = [os.path.basename(json_file) for json_file in self.json_list]
        for lidar_file in self.lidar_list.copy():
            lidar_base = os.path.basename(lidar_file)
            lidar_json = lidar_base.split(".")[0] + ".json"
            if lidar_json in laszy_json_bases:
                self.lidar_list.remove(lidar_file)

    def write(self, name: str = "", validate=False, check_logs: bool = True):

        """
        Write list of LAS/LAZ file summaries to a csv file.

        Static method that accepts a list of LAS/LAZ files and writes
        their respective summaries to rows in a csv file.

        :param validate: When True, function will call validate_report() to output lidar error reports.
        :param check_logs: Check for existing completed logs to ignore previously processed files.
        :param name: Output filename (default='laszy_report.csv')
        """

        existing_data = ""
        if not bool(name):
            name = self._DEFAULT_NAME

        if not name.endswith(".csv"):
            name += ".csv"

        self._path = os.path.join(self.outdir, name)
        if check_logs:
            existing_data = self.__check_logs(existing_data, self._path)

        with open(self._path, "w") as csv:
            self.__write_report(csv, existing_data)

        if validate:
            self.validate_report()

        for is_lidar in [True, False]:
            self.__log_completed(lidar=is_lidar)

        self.__write_err(self._path)

    def validate_report(self, path: str = "", outdir=""):

        if not bool(path):
            path = self._path

        if os.path.exists(path):
            issues = {}
            df = pd.read_csv(path)

            df = df.drop(LASZY_REPORT_DROP_COLUMNS, axis=1)

            df = self.__public_header_check(df, issues)
            df = self.__xyx_scale_check(df, issues)
            df = self.__xyz_offset_check(df, issues)
            df = self.__global_encoding_check(df, issues)
            df = self.__crs_check(df, issues)
            df = self.__point_records_check(df, issues)

            if bool(issues):
                issues = {key: int(issues[key]) for key, value in issues.items()}
                _outdir = os.path.dirname(path) if not bool(outdir) else outdir
                name = os.path.basename(path)
                name_no_ext = name.split(".")[0]

                # write the json summary
                out_summary_name = name_no_ext + "_errors_summary.json"
                with open(os.path.join(_outdir, out_summary_name), "w") as json_summary:
                    json.dump(issues, json_summary, indent=4)

                # write the errors file
                out_csv_name = os.path.join(_outdir, name_no_ext + "_errors.csv")
                df.to_csv(out_csv_name)

    def __write_report(self, csv, existing_data):

        """
        Write final CSV report.

        Private helper method.

        :param csv: Open csv file-like object.
        :param existing_data: Previous CSV data.
        """

        if bool(existing_data):
            csv.write(existing_data)
        else:
            csv.write(",".join(_LaszyReportColumns.COLUMNS) + "\n")
        self.__from_lidar_list(csv)
        self.__from_json_list(csv)

    def __check_logs(self, existing_data, out):

        """
        Check existing log files and inherit data from previous reports (if possible).

        :param existing_data: existing data from previous csv file.
        :param out: out directory.
        """

        for is_lidar in [False, True]:
            self.__check_for_completed(lidar=is_lidar)
        if os.path.exists(out):
            with open(out, "r") as f:
                existing_data = f.read()
        return existing_data

    def __log_completed(self, lidar: bool = False):

        """
        Write a list of processed files to a log file.

        :param lidar: When True, will write LiDAR log, otherwise, will write JSON log.
        """

        completed = self._lidar_completed if lidar else self._json_completed
        out_name = self._LIDAR_LOG_NAME if lidar else self._JSON_LOG_NAME

        if bool(completed):
            with open(os.path.join(self.outdir, out_name), "w") as f:
                for file in completed:
                    f.write(file + "\n")

    def __write_err(self, out):

        """
        Write any errors encountered to log file.

        :param out: Output report name.
        """

        if bool(self._errors):
            with open(f"{out}_exceptions.log", "w") as err_log:
                for err in self._errors:
                    fname = err[0]
                    exception = err[1]
                    err_log.write(fname + f"\n\t{exception}\n")

    def __from_lidar_list(self, csv):

        """
        Write rows ro open CSV file from list of lidar files (LAS/LAZ).

        :param csv: Open file pointer to CSV file object.
        """

        if bool(self.lidar_list):
            json_outdir = os.path.join(self.outdir, "laszy_json") if self.las_to_json else ""
            files = tqdm.tqdm(self.lidar_list, desc="Processing LAS/LAZ files...") if self.verbose else self.lidar_list
            for file in files:
                las = Laszy.Laszy(file)
                try:
                    s = las.summarize(outdir=json_outdir)
                    row = self.__get_row(s)
                    csv.write(",".join(row) + "\n")
                    self._lidar_completed.append(file)

                except Exception as e:
                    is_possibly_corrupt = (not bool(las.public_header_block))
                    self._errors.append(
                        (file, (CORRUPT_FILE_MSG if is_possibly_corrupt else e) + "\n")
                    )

    def __from_json_list(self, csv):

        """
        Write rows to CSV file from list of json files.

        :param csv: Open file pointer to CSV file object.
        """

        if bool(self.json_list):
            files = tqdm.tqdm(self.json_list, desc="Processing JSON files...") if self.verbose else self.json_list
            for file in files:
                try:
                    with open(file, "r") as f:
                        summary = json.load(f)
                        row = self.__get_row(summary)
                        csv.write(",".join(row) + "\n")
                    self._json_completed.append(file)

                except Exception as e:
                    self._errors.append((file, e))

    def __check_for_completed(self, lidar: bool = False):

        """
        Check for existing logs to see if files should be skipped.

        Checks in self.outdir directory for 'lidar_completed.log' or
        'json_completed.log'. If either of these files exist, the contents
        will be compared with 'self.las_list' and 'self.json_list'.

        If filenames are present in both the log files and the completed logs,
        these files will be removed from processing.

        This feature is implemented in an effort to keep the utility from
        executing files that have already been processed.

        :param lidar: When True, will check lidar logs, otherwise, will check json logs.
        """

        file_list = self.lidar_list if lidar else self.json_list
        log_name = self._LIDAR_LOG_NAME if lidar else self._JSON_LOG_NAME

        log = os.path.join(self.outdir, log_name)
        if os.path.exists(log):
            with open(log, "r") as f:
                contents = f.read()
                ignore_list = contents.split("\n")

            set_ignore = set(ignore_list)
            set_file_list = set(file_list)
            file_list_ = list(set_file_list.difference(set_ignore))

            if lidar:
                self.lidar_list = file_list_
            else:
                self.json_list = file_list_

    @staticmethod
    def __get_row(summary: dict) -> list[str]:

        """
        Get a single row for output csv.

        Uses the keys stored in enum class '_LaszyReportColumns' to
        retrieve the values stored in a json dictionary that represents
        a Laszy summary and casts all values to a string.

        :param summary: Dictionary object containing laszy summary data.
        :return: List of strings containing each values in laszy summary.
        """

        pr = summary["point_records"]
        phb = summary["public_header_block"]
        pub_hdr_vals = [str(phb[key]) for key in _LaszyReportColumns.PUB_HDR]
        ge_vals = [str(phb["global_encoding"][key]) for key in _LaszyReportColumns.GLOBAL_ENCODING]
        crs_vals = [str(summary["crs"][key]) for key in _LaszyReportColumns.CRS]
        vlr_vals = [str(summary["vlrs"][key]) for key in _LaszyReportColumns.VLR_HDR]
        point_vals = [str(pr[key]) for key in _LaszyReportColumns.POINT_RECORDS]
        evlr_vals = [str(summary["evlrs"][key]) for key in _LaszyReportColumns.EVLR_HDR]

        flag_vals = [
            (str(pr["class_flags"][key]) if bool(pr["class_flags"]) else "N/A")
            for key in _LaszyReportColumns.CLASS_FLAGS
        ]

        row = [
            summary["filename"], *pub_hdr_vals, *ge_vals, *crs_vals,
            *vlr_vals, *point_vals, *flag_vals, *evlr_vals, str(summary["rgb_encoding"]), summary["wkt_bbox"]
        ]

        # make sure to wrap each row item in quotes
        # if the item contains a csv seperator in it
        for i in range(len(row)):
            if row[i].find(",") >= 0:
                row[i] = f"\"{row[i]}\""

        return row

    @staticmethod
    def __global_encoding_check(df, issues):

        """Check global encoding value for issues"""

        df['global_encoding'] = df['global_encoding'].apply(LaszyReport.__is_globalencoding_invalid)
        col = df["global_encoding"] != ""
        if col.sum() > 0:
            issues.update({"global_encoding_value": col.sum()})
        else:
            df = df.drop("global_encoding", axis=1)

        df['wkt_crs'] = df['wkt_crs'].apply(LaszyReport.__is_wktflag_invalid)
        col = df["wkt_crs"] != ""
        if col.sum() > 0:
            issues.update({"wkt_crs_flag": col.sum()})
        else:
            df = df.drop("wkt_crs", axis=1)

        df['gps_standard_time'] = df['gps_standard_time'].apply(LaszyReport.__is_gpstimeflag_invalid)
        col = df["gps_standard_time"] != ""
        if col.sum() > 0:
            issues.update({"gps_time_flag": col.sum()})
        else:
            df = df.drop("gps_standard_time", axis=1)
        # guid contract number check
        df['synthetic_returns'] = df['synthetic_returns'].apply(LaszyReport.__is_syntheticflag_invalid)
        col = df["synthetic_returns"] != ""
        if col.sum() > 0:
            issues.update({"synthetic_returns_flag": col.sum()})
        else:
            df = df.drop("synthetic_returns", axis=1)

        return df

    @staticmethod
    def __xyz_offset_check(df, issues):

        """Check XYZ offset for issues."""

        df['x_offset'] = df['x_offset'].apply(LaszyReport.__is_xoffset_invalid)
        col = df["x_offset"] != ""
        if col.sum() > 0:
            issues.update({"x_offset": col.sum()})
        else:
            df = df.drop("x_offset", axis=1)

        df['y_offset'] = df['y_offset'].apply(LaszyReport.__is_yoffset_invalid)
        col = df["y_offset"] != ""
        if col.sum() > 0:
            issues.update({"y_offset": col.sum()})
        else:
            df = df.drop("y_offset", axis=1)

        df['z_offset'] = df['z_offset'].apply(LaszyReport.__is_zoffset_invalid)
        col = df["z_offset"] != ""
        if col.sum() > 0:
            issues.update({"z_offset": col.sum()})
        else:
            df = df.drop("z_offset", axis=1)

        return df

    @staticmethod
    def __xyx_scale_check(df, issues):

        """Check XYZ scaling for issues."""

        # guid contract number check
        df['x_scale'] = df['x_scale'].apply(LaszyReport.__is_xscale_invalid)
        col = df["x_scale"] != ""
        if col.sum() > 0:
            issues.update({"x_scale": col.sum()})
        else:
            df = df.drop("x_scale", axis=1)

        # guid contract number check
        df['y_scale'] = df['y_scale'].apply(LaszyReport.__is_yscale_invalid)
        col = df["y_scale"] != ""
        if col.sum() > 0:
            issues.update({"y_scale": col.sum()})
        else:
            df = df.drop("y_scale", axis=1)

        # guid contract number check
        df['z_scale'] = df['z_scale'].apply(LaszyReport.__is_zscale_invalid)
        col = df["z_scale"] != ""
        if col.sum() > 0:
            issues.update({"z_scale": col.sum()})
        else:
            df = df.drop("z_scale", axis=1)

        return df

    @staticmethod
    def __point_records_check(df, issues):

        """Check point record fields for any issues with data."""

        # check for class code 0
        df['classes'] = df['classes'].apply(LaszyReport.__is_neverclassified_points)
        col = df["classes"] != ""
        if col.sum() > 0:
            issues.update({"points_in_never_classified": col.sum()})
        else:
            df = df.drop("classes", axis=1)

        # check for invalid flightline numbers
        df['flightline_start'] = df['flightline_start'].apply(LaszyReport.__is_flightlines_invalid)
        col = df["flightline_start"] != ""
        if col.sum() > 0:
            issues.update({"invalid_flightline_numbers": col.sum()})
        else:
            df = df.drop(["flightline_start", "flightline_end"], axis=1)

        # check for invalid gps times
        df['gps_time_min'] = df['gps_time_min'].apply(LaszyReport.__is_gpsweektime_present)
        col = df["gps_time_min"] != ""
        if col.sum() > 0:
            df = df.drop("gps_time_max", axis=1)
            issues.update({"gps_week_time_found": col.sum()})
        else:
            df = df.drop(["gps_time_min", "gps_time_max"], axis=1)

        # check for synthetic flags
        df['has_synthetic'] = df['has_synthetic'].apply(LaszyReport.__is_syntheticclassflag_invalid)
        col = df["has_synthetic"] != ""
        if col.sum() > 0:
            issues.update({"synthetic_class_flags": col.sum()})
        else:
            df = df.drop("has_synthetic", axis=1)

        # check if no wkt crs is present at all
        df['date_end'] = df['date_end'].apply(LaszyReport.__is_date_from_future)
        col = df["date_end"] != ""
        if col.sum() > 0:
            df["invalid_dates"] = col
            issues.update({"invalid_dates_found": col.sum()})
        df = df.drop('date_end', axis=1)

        return df

    @staticmethod
    def __crs_check(df, issues):

        """Check CRS for any issues."""

        # check for existence of compound crs
        df['compd_cs'] = df['compd_cs'].apply(LaszyReport.__is_compdcs_invalid)
        col = df["compd_cs"] != ""
        if col.sum() > 0:
            issues.update({"compd_cs": col.sum()})
        else:
            df = df.drop('compd_cs', axis=1)

        # check the vertical datum
        df['vert_datum'] = df['vert_datum'].apply(LaszyReport.__is_vertdatum_invalid)
        col = df["vert_datum"] != ""
        if col.sum() > 0:
            issues.update({"vert_datum": col.sum()})
        else:
            df = df.drop('vert_datum', axis=1)

        # check the horizontal datum
        df['hz_datum'] = df['hz_datum'].apply(LaszyReport.__is_hzdatum_invalid)
        col = df["hz_datum"] != ""
        if col.sum() > 0:
            issues.update({"hz_datum": col.sum()})
        else:
            df = df.drop('hz_datum', axis=1)

        # check if no wkt crs is present at all
        df['vlr_has_wkt_crs'] = df['vlr_has_wkt_crs'].apply(LaszyReport.__is_vlrwkt_empty)
        df['evlr_has_wkt_crs'] = df['evlr_has_wkt_crs'].apply(LaszyReport.__is_vlrwkt_empty)
        col_vlr = df["vlr_has_wkt_crs"] != ""
        col_evlr = df["evlr_has_wkt_crs"] != ""
        col = col_vlr & col_evlr
        if ~col.sum() > 0:
            df["no_wkt_found"] = col
            issues.update({"vlr_has_wkt_crs": col.sum()})
        df = df.drop(['vlr_has_wkt_crs', 'evlr_has_wkt_crs'], axis=1)

        return df

    @staticmethod
    def __public_header_check(df, issues):

        # guid contract number check
        df['guid_asc'] = df['guid_asc'].apply(LaszyReport.__is_contract_invalid)
        col = df["guid_asc"] != ""
        if col.sum() > 0:
            issues.update({"guid_contract_number": col.sum()})
        else:
            df = df.drop("guid_asc", axis=1)

        # System ID format check
        df['system_id'] = df['system_id'].apply(LaszyReport.__is_systemid_invalid)
        col = df["system_id"] != ""
        if col.sum() > 0:
            issues.update({"system_id_format": col.sum()})
        else:
            df = df.drop("system_id", axis=1)

        # version check
        df['version'] = df['version'].apply(LaszyReport.__is_lasversion_invalid)
        col = df["version"] != ""
        if col.sum() > 0:
            issues.update({"version": col.sum()})
        else:
            df = df.drop("version", axis=1)

        # Point data record format check
        df['point_data_format'] = df['point_data_format'].apply(LaszyReport.__is_pointformat_invalid)
        col = df["point_data_format"] != ""
        if col.sum() > 0:
            issues.update({"point_data_format": col.sum()})
        else:
            df = df.drop("point_data_format", axis=1)

        if ACQUISITON:
            # File source id vs filename number check
            df['filename_has_correct_source_id'] = df.apply(LaszyReport.__is_sourceid_valid, axis=1)
            col = df['filename_has_correct_source_id'] != ""
            if col.sum() > 0:
                issues.update({"filename_has_correct_source_id": col.sum()})
            else:
                df = df.drop("filename_has_correct_source_id", axis=1)

        return df

    @staticmethod
    def __is_sourceid_valid(row):
        numb5 = str(row["filename"]).split("_")[0]
        fsid = str(row["file_source_id"])
        return "Correct" if fsid == numb5 else "Filename does not contain File Source ID"

    @staticmethod
    def __is_syntheticclassflag_invalid(synthetic_class_flag: bool):
        expected = False
        if not math.isnan(synthetic_class_flag) and synthetic_class_flag != expected:
            return synthetic_class_flag
        else:
            return ""

    @staticmethod
    def __is_gpsweektime_present(gps_min_time: int):
        max_gps_week_time = 604800
        if max_gps_week_time >= gps_min_time:
            return str(gps_min_time)
        else:
            return ""

    @staticmethod
    def __is_flightlines_invalid(point_source_id: int):
        min_valid_fl = 1
        if point_source_id < min_valid_fl:
            return str(point_source_id)
        else:
            return ""

    @staticmethod
    def __is_neverclassified_points(classes: str):
        expected = 0
        classes = classes.replace("[", "").replace("]", "")
        classes = [int(val) for val in classes.split(",")]
        if expected in classes:
            return str(classes)
        else:
            return ""

    @staticmethod
    def __is_vlrwkt_empty(vlr_has_wkt: bool):
        expected = True
        if vlr_has_wkt != expected:
            return vlr_has_wkt
        else:
            return ""

    @staticmethod
    def __is_hzdatum_invalid(hz_datum: str):
        expected = "NAD83_Canadian_Spatial_Reference_System"
        if hz_datum != expected:
            return hz_datum
        else:
            return ""

    @staticmethod
    def __is_vertdatum_invalid(vert_datum: str):
        expected = "Canadian Geodetic Vertical Datum of 2013"
        if vert_datum != expected:
            return vert_datum
        else:
            return ""

    @staticmethod
    def __is_compdcs_invalid(compdcs: str):
        expected = bool(compdcs)
        if not expected:
            return "No compound projection"
        else:
            return ""

    @staticmethod
    def __is_contract_invalid(guid: str):
        expected = re.compile(RegexLidar.CONTRACT_NUMBER)
        if bool(guid):
            if (not isinstance(guid, str)) and (not isinstance(guid, bytes)):
                return "unknown format"
            elif not bool(expected.search(guid)):
                return guid
            else:
                return ""
        return "No GUID found"

    @staticmethod
    def __is_systemid_invalid(system_id: str):
        expected = re.compile(RegexLidar.SYSTEM_ID_PRODUCTION)
        if bool(system_id):
            if (not isinstance(system_id, str)) and (not isinstance(system_id, bytes)):
                return "unknown format"
            elif not bool(expected.search(system_id)):
                return system_id
            else:
                return ""
        return "No System ID found"

    @staticmethod
    def __is_lasversion_invalid(lasversion: float):
        expected = 1.4
        if lasversion != expected:
            return str(lasversion)
        else:
            return ""

    @staticmethod
    def __is_pointformat_invalid(pointformat: int):
        expected = 6
        if pointformat != expected:
            return str(pointformat)
        else:
            return ""

    @staticmethod
    def __is_xscale_invalid(xscale: float):
        expected = 0.01
        if xscale != expected:
            return str(xscale)
        else:
            return ""

    @staticmethod
    def __is_yscale_invalid(yscale: float):
        expected = 0.01
        if yscale != expected:
            return str(yscale)
        else:
            return ""

    @staticmethod
    def __is_zscale_invalid(zscale: float):
        expected = 0.01
        if zscale != expected:
            return str(zscale)
        else:
            return ""

    @staticmethod
    def __is_xoffset_invalid(xoffset: float):
        expected = 0
        modulo = float(xoffset) % 1
        if modulo != expected:
            return str(xoffset)
        else:
            return ""

    @staticmethod
    def __is_yoffset_invalid(yoffset: float):
        expected = 0
        modulo = float(yoffset) % 1
        if modulo != expected:
            return str(yoffset)
        else:
            return ""

    @staticmethod
    def __is_zoffset_invalid(zoffset: float):
        expected = 0
        modulo = float(zoffset) % 1
        if modulo != expected:
            return str(zoffset)
        else:
            return ""

    @staticmethod
    def __is_globalencoding_invalid(value: int):
        expected = 17
        if value != expected:
            return str(value)
        else:
            return ""

    @staticmethod
    def __is_wktflag_invalid(flag: bool):
        expected = True
        if flag != expected:
            return flag
        else:
            return ""

    @staticmethod
    def __is_gpstimeflag_invalid(flag: bool):
        expected = True
        if flag != expected:
            return flag
        else:
            return ""

    @staticmethod
    def __is_syntheticflag_invalid(flag: bool):
        expected = False
        if flag != expected:
            return flag
        else:
            return ""

    @staticmethod
    def __is_date_from_future(date: str):
        today = datetime.datetime.today()
        in_date = date.split(" ")[0]
        ymd = in_date.split("-")

        # first check if GpsWeekTimeError string is present
        is_invalid_date = date == Laszy.GPS_WEEK_TIME_ERR_STR
        if not is_invalid_date:  # now check the actual date
            year, month, day = int(ymd[0]), int(ymd[1]), int(ymd[2])
            invalid_year = today.year < year
            invalid_month = (today.year == year) and (today.month < month)
            invalid_day = (today.year == year) and (today.month == month) and (today.day < day)
            is_invalid_date = invalid_day or invalid_month or invalid_year
        if is_invalid_date:
            return date
        else:
            return ""
