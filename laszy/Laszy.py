import os
import re
import io
import sys
import glob
import uuid
import math
import json
import tqdm
import laspy
import datetime
import numpy as np
import pandas as pd
from typing import Union
from lazrs import LazrsError
from collections import namedtuple

from rsge_toolbox.util import WktCrsInfo
from rsge_toolbox.util import time_tools
from rsge_toolbox.util.WktCrsInfo import WktCrsInfo
from rsge_toolbox.lidar.lidar_const import ASPRS, RegexLidar


# ------------------------------------
# -- Type Definitions
# ------------------------------------
PointFilterType = namedtuple("PointFilterType", "LAST_RETURN IGNORE_CLASS IGNORE_RETURN")


# ------------------------------------
# -- Misc. Constants
# ------------------------------------
GPS_WEEK_TIME_LENGTH = 6
GPS_WEEK_TIME_ERR_STR = "GpsDateConversionError"

# ------------------------------------
# -- Encoding Constants
# ------------------------------------
UTF_8 = "utf-8"
UNICODE_DECODE_ERROR = "UnicodeDecodeError"


# ------------------------------------
# -- LAS/LAZ related constants
# ------------------------------------
POINT_FILTER_TYPE = PointFilterType(
    LAST_RETURN=0, IGNORE_RETURN=-1, IGNORE_CLASS=-1
)


# =========================================================
# --- Laszy class
# =========================================================
class Laszy:

    """
    Class for reading, parsing, and interpreting LAS/LAZ data.

    Attributes:
        - public_header_block: PublicHeaderBlock object.
        - vlrs: List of VariableLengthRecord objects.
        - points: Laspy.LasData object (optional) (initialized to None).
        - evlrs: List of VariableLengthRecord objects.

    """

    def __init__(self, file: str, read_points=True):

        if not self.__is_lidar_file(file):
            raise NotLidarFileError

        self.file_basename = os.path.basename(file) if bool(file) else ""
        self.file_absolute = file if bool(file) else ""
        reader = laspy.read if read_points else laspy.open

        try:
            self._lasdata = reader(file, laz_backend=laspy.LazBackend.LazrsParallel)
            self.public_header_block = self._lasdata.header
            self.vlrs = self._lasdata.header.vlrs
            self.points = self._lasdata.points if read_points else None
            self.evlrs = self._lasdata.evlrs
        except LazrsError:
            self._lasdata = None
            self.public_header_block = None
            self.vlrs = None
            self.points = None
            self.evlrs = None

    def read_points(self):

        """
        Call laspy.read() to get point record data.
        """

        file = self.file_absolute

        if bool(file):
            self._lasdata.seek(0, io.SEEK_SET)
            self.points = self._lasdata.read_points(self.public_header_block.point_count)

    def set_lasdata(self, lasdata: Union[laspy.LasReader, laspy.LasData]):

        """
        Set self._lasdata using Laspy object.

        Sets the _lasdata property, initializing all related properties that
        depend on _lasdata. This method is particularly useful when a user
        wishes to utilize the functionality of laszy using an already
        initialized laspy object.

        Note that:
            If calling set_lasdata with a LasReader object, (with laspy.open()),
            the points attribute will not be assigned.

            If calling set_lasdata with a LasData object, (with laspy.read()),
            the points attribute will be assigned.

        :param lasdata: LasReader or LasData object.
        """

        has_points = isinstance(lasdata, laspy.LasData)

        self.file_basename = ""
        self.file_absolute = ""
        self._lasdata = lasdata
        self.public_header_block = self._lasdata.header
        self.vlrs = self._lasdata.header.vlrs
        self.points = self._lasdata.points if has_points else None
        self.evlrs = self._lasdata.evlrs

    def get_classes(self) -> list[int]:

        """
        Get point classes present in data.

        :return: List of classes present in all point records.
        """

        classes = list(np.unique(self._lasdata.classification))
        classes = [int(val) for val in classes]

        return classes

    def filter_points(self, class_num: int = POINT_FILTER_TYPE.IGNORE_CLASS, return_num: int = POINT_FILTER_TYPE.IGNORE_RETURN) -> Union[laspy.ScaleAwarePointRecord, None]:

        """
        Filter point cloud by return number or class number.

        Returns a numpy ndarray object containing points filtered by input params.
        By default, both class_num and return_num are set to IGNORE.

        NOTE: return_num=0 indicates LAST_RETURN
        NOTE: return_num=-1 indicates IGNORE
        NOTE: class_num=-1 indicates IGNORE

        :param return_num: Integer value indicating return number.
        :param class_num: Integer value indicating class number.

        :return: Filtered np.ndarray of points.
        """

        filtered = None
        las = self._lasdata

        if not bool(las):
            return filtered

        return_filter = return_num

        should_filter_return = return_num > POINT_FILTER_TYPE.IGNORE_RETURN
        should_filter_class = class_num > POINT_FILTER_TYPE.IGNORE_CLASS

        if should_filter_return:
            return_filter = self.points.num_returns if return_num == POINT_FILTER_TYPE.LAST_RETURN else return_num

        if should_filter_class and should_filter_return:
            filtered = las.points[return_filter == las.return_num & las.classification == class_num]

        elif should_filter_return:
            filtered = las.points[return_filter == las.return_num]

        elif should_filter_class:
            filtered = las.points[las.classification == class_num]

        return filtered

    def get_density(self, class_num: int = POINT_FILTER_TYPE.IGNORE_CLASS, return_num: int = POINT_FILTER_TYPE.IGNORE_RETURN) -> float:

        """
        Calculate density of the LAS/LAZ data.

        Computes point density based on X/Y geographic bounds of the data.
        User may filter by return or class number.

        By default, both class_num and return_num are set to -1. When
        either of these arguments are set to a value LESS THAN 0, these
        filters are ignored.

        Additionally, a 'return_num' of 0 indicates LAST_RETURN, and will
        hence filter the point by last return. Moreover, LAST_RETURN is also
        provided as a constant equal to the value 0.

        :param class_num: Return number to filter points by. (0 = LAST_RETURN; N < 0 = IGNORE)
        :param return_num: Return number to filter points by. (0 = LAST_RETURN; N < 0 = IGNORE)
        :return: Point density for entire point cloud as float or -1 if failure.
        """

        density = -1  # return invalid density if failure as a signal something went wrong

        min_x, max_x = self.get_x_minmax()
        min_y, max_y = self.get_y_minmax()
        dim_x, dim_y = max_x - min_x, max_y - min_y

        filtered = self.filter_points(class_num, return_num)
        if bool(filtered):
            density = filtered.array.size/(dim_x * dim_y)

        return density

    def get_global_encoding(self, value_only: bool = False) -> Union[dict, int]:

        """
        Interpret the Global Encoding value in the LAS/LAZ data.

        Reads the Global Encoding and performs bitwise checks for flags:
            - gps time type
            - internal waveform packets
            - external waveform packets
            - has synthetic returns.
            - CRS info type

        Note that a value of False in the resulting dictionary means
        that the bit associated with the dictionary key is set to 0.

        :return: Dictionary containing the point flag name, and a boolean indicating the status of the bit field.
        :return: Integer value of global encoding (if value_only=True).
        :return: None if failure.
        """

        ge = self.public_header_block.global_encoding

        if value_only:
            return ge.value

        ged = {
            "global_encoding": ge.value,
            "gps_standard_time": bool(ge.gps_time_type),
            "waveform_internal_packets": ge.waveform_data_packets_internal,
            "waveform_external_packets": ge.waveform_data_packets_external,
            "synthetic_returns": ge.synthetic_return_numbers,
            "wkt_crs": ge.wkt
        }

        return ged

    def get_crs_info(self) -> str:

        """
        Get CRS info in VLRs or EVLRs (if present).

        Returns the CRS info of the LAS/LAZ data as a single string.

        :return: Entire CRS string from VLRs if present
        :return: Empty string if no info present.
        """

        # check the VLRs
        vlr = self.vlrs.get("WktCoordinateSystemVlr") if bool(self.vlrs) else None
        if bool(vlr):
            return vlr[0].string

        # check the VLRs
        evlr = self.evlrs.get("WktCoordinateSystemVlr") if bool(self.evlrs) else None
        if bool(evlr):
            return evlr[0].string

        return ""  # will return an empty string if no WKT CRS in VLRs

    def get_x_minmax(self):

        """
        Get LasHeader min/max values.

        :return: Tuple (min, max) on success.
        :return: Tuple (None, None) on failure.
        """

        pub_hdr = self.public_header_block

        return pub_hdr.x_min, pub_hdr.x_max

    def get_y_minmax(self):

        """
        Get LasHeader min/max values.

        :return: Tuple (min, max) on success.
        :return: Tuple (None, None) on failure.
        """

        pub_hdr = self.public_header_block

        return pub_hdr.y_min, pub_hdr.y_max

    def get_z_minmax(self):

        """
        Get LasHeader min/max values.

        :return: Tuple (min, max) on success.
        :return: Tuple (None, None) on failure.
        """

        return self.public_header_block.z_min, self.public_header_block.z_max

    def get_guid_hex(self) -> str:

        """
        Combine guids 1-4 into a single hexadecimal string.

        :return: string representing guids 1-4 in hexadecimal encoding.
        """

        pub_hdr = self.public_header_block

        first = pub_hdr.uuid.hex[0:8]
        second = pub_hdr.uuid.hex[8:12]
        third = pub_hdr.uuid.hex[12:16]
        fourth = pub_hdr.uuid.hex[16:20]
        fifth = pub_hdr.uuid.hex[20:32]

        fourth, fifth = self.__swap_guid_chars(fourth, fifth)

        return "-".join([first, second, third, fourth, fifth])

    def get_gps_time_minmax(self) -> tuple:

        """
        Get the minimum and maximum GPS times for the LAS/LAZ data.

        :return: tuple -> (min_time, max_time)
        """

        gps_times = self.points.gps_time
        gps_min = np.min(gps_times)
        gps_max = np.max(gps_times)

        return float(gps_min), float(gps_max)

    def get_point_source_id_minmax(self) -> tuple:

        """
        Get the minimum and maximum point source IDs for the LAS/LAZ data.

        :return: tuple -> (min_psid, max_psid)
        """

        pt_src_ids = self.points.pt_src_id
        psid_min = np.min(pt_src_ids)
        psid_max = np.max(pt_src_ids)

        return int(psid_min), int(psid_max)

    def get_classification_flags(self) -> Union[dict, None]:

        """
        Get classification flags present in point records (if any).

        Reads the point records and performs bitwise checks for flags:
            - synthetic
            - withheld
            - keypoint
            - overlap

        Note that this utility will only report if these flags are present
        IN ANY of the point records.

        :return: Dictionary containing the point flag name, and a boolean indicating the status of the bit field.
        :return: None if failure.
        """

        point_flags = None

        # classification flags only exist for point formats 6-10
        pdr = self._lasdata.point_format.id
        class_flags_exist = (6 <= pdr <= 10)
        if bool(pdr) and not class_flags_exist:
            return point_flags

        class_flags = np.sort(self.points.classification_flags)
        check_synthetic = np.bitwise_and(class_flags, ASPRS.ClassFlag.SYNTHETIC)
        check_withheld = np.bitwise_and(class_flags, ASPRS.ClassFlag.WITHHELD)
        check_keypoint = np.bitwise_and(class_flags, ASPRS.ClassFlag.KEYPOINT)
        check_overlap = np.bitwise_and(class_flags, ASPRS.ClassFlag.OVERLAP)

        point_flags = {  # check if a flagged point exists after bitwise-and
            "has_synthetic": ASPRS.ClassFlag.SYNTHETIC in check_synthetic,
            "has_keypoint": ASPRS.ClassFlag.KEYPOINT in check_keypoint,
            "has_withheld": ASPRS.ClassFlag.WITHHELD in check_withheld,
            "has_overlap": ASPRS.ClassFlag.OVERLAP in check_overlap
        }

        return point_flags

    def is_rgb_encoded(self) -> bool:

        """
        Check if point record format ID contains rgb encoding fields.

        :return: True or False
        """

        rgb_record_ids = [2, 3, 5, 7, 8, 10]
        pid = self.public_header_block.point_format.id

        return pid in rgb_record_ids

    def get_version(self) -> str:

        """
        Get the Major and Minor version of the inpuit LAS/LAZ data.

        :return: string of the LAS/LAZ version (e.g '1.4')
        """

        version = ""
        if not bool(self.file_basename):
            return version

        pub_hdr = self.public_header_block
        version = f"{pub_hdr.major_version}.{pub_hdr.minor_version}"

        return version

    def get_wkt_boundingbox(self) -> str:

        """
        Get bounding box of input LAS/LAZ data as WKT POLYGON string.

        :return: WKT POLYGON string.
        """

        x_min, x_max = self.get_x_minmax()
        y_min, y_max = self.get_y_minmax()

        p_ll = f"{x_min} {y_min}"
        p_ul = f"{x_min} {y_max}"
        p_ur = f"{x_max} {y_max}"
        p_lr = f"{x_max} {y_min}"

        wkt_str = f"POLYGON(({p_ll}, {p_ul}, {p_ur}, {p_lr}, {p_ll}))"

        return wkt_str

    def vlrs_have_wkt_crs(self, evlr: bool = False) -> bool:

        """
        Check if VLRs has crs info in them.

        :return: bool
        """

        records = self.evlrs if evlr else self.vlrs
        vlr = records.get("WktCoordinateSystemVlr") if bool(records) else None

        if bool(vlr):
            return True

        return False

    def vlrs_have_geotiff_crs(self, evlr: bool = False) -> bool:

        """
        Check if VLRs has crs info in them.

        :return: bool
        """

        records = self.evlrs if evlr else self.vlrs
        if bool(records):
            has_geo_double = bool(records.get("GeoDoubleParamsVlr"))
            has_geo_ascii = bool(records.get("GeoAsciiParamsVlr"))
            has_geo_key = bool(records.get("GeoKeyDirectoryVlr"))
            if has_geo_key or has_geo_ascii or has_geo_double:
                return True

        return False

    def summarize(self, header_only=False, outdir="") -> Union[dict, None]:

        """
        Summarize the input LAS/LAZ data into a dictionary.

        User may optionally pass header_only=True (False by default).
        When header_only is set to True, this utility will omit the
        point data. This is considerably faster, however certain data
        that can only be derived from the point records will not be
        present in the result.

        If 'outdir' is NOT an empty string, function will write results to
        file in JSON format. Note that even if the 'outdir' is not valid,
        function will create a directory to write hte results to.

        :param outdir: Out directory. If provided, results will also be writting to a file ({self._file}.json)
        :param header_only: boolean value. Determines whether to read the point data.
        :return:
        """

        point_record_summary = None
        pub_hdr = self.public_header_block
        if not header_only and pub_hdr.point_count > 0:
            point_record_summary = self.__point_record_summary()

        summary = {
            "filename": self.file_basename,
            "public_header_block": self.__public_header_summary(),
            "crs": self.__crs_info_summary(),
            "vlrs": {
                "vlr_count": len(pub_hdr.vlrs),
                "vlr_has_wkt_crs": self.vlrs_have_wkt_crs(),
                "vlr_has_geotiff_crs": self.vlrs_have_geotiff_crs(),
                "records": self.__vlr_summary()
            },
            "point_records": point_record_summary,
            "evlrs": {
                "evlr_count": pub_hdr.number_of_evlrs,
                "evlr_has_wkt_crs": self.vlrs_have_wkt_crs(evlr=True),
                "evlr_has_geotiff_crs": self.vlrs_have_geotiff_crs(evlr=True),
                "records": self.__vlr_summary(evlr=True)
            },
            "rgb_encoding": self.is_rgb_encoded(),
            "wkt_bbox": self.get_wkt_boundingbox()
        }

        if bool(outdir):
            self.__summary_to_json(outdir, summary)

        return summary

    def __public_header_summary(self) -> dict:

        """
        Summarize the input LAS/LAZ public header block data into a dictionary.

        :return:
        """

        pub_hdr = self.public_header_block
        x_min, x_max = self.get_x_minmax()
        y_min, y_max = self.get_y_minmax()
        z_min, z_max = self.get_z_minmax()
        x_dec_places = str(pub_hdr.x_scale)[::-1].find('.')
        y_dec_places = str(pub_hdr.y_scale)[::-1].find('.')
        z_dec_places = str(pub_hdr.z_scale)[::-1].find('.')

        pub_hdr_summary = {
            "global_encoding": self.get_global_encoding(),
            "guid_asc": self.get_guid_asc(),
            "guid_hex": self.get_guid_hex(),
            "file_source_id": pub_hdr.file_source_id,
            "system_id": pub_hdr.system_identifier,
            "generating_software": pub_hdr.generating_software,
            "creation_date": self.__format_creation_date(pub_hdr),
            "version": self.get_version(),
            "point_data_format": pub_hdr.point_format.id,
            "point_count": pub_hdr.point_count,
            "x_min": round(x_min, x_dec_places),
            "x_max": round(x_max, x_dec_places),
            "y_min": round(y_min, y_dec_places),
            "y_max": round(y_max, y_dec_places),
            "z_min": round(z_min, z_dec_places),
            "z_max": round(z_max, z_dec_places),
            "x_scale": pub_hdr.x_scale,
            "y_scale": pub_hdr.y_scale,
            "z_scale": pub_hdr.z_scale,
            "x_offset": round(pub_hdr.x_offset, x_dec_places),
            "y_offset": round(pub_hdr.y_offset, y_dec_places),
            "z_offset": round(pub_hdr.z_offset, z_dec_places),
        }

        return pub_hdr_summary

    def __point_record_summary(self) -> dict:

        """
        Summarize the input LAS/LAZ point data into a dictionary.

        :return:
        """

        gps_min, gps_max = self.get_gps_time_minmax()
        fl_min, fl_max = self.get_point_source_id_minmax()
        gps_min_week_time = self.__is_gps_week_time(gps_min)
        gps_max_week_time = self.__is_gps_week_time(gps_max)
        point_records_summary = {
            "classes": self.get_classes(),
            "gps_time_min": gps_min,
            "gps_time_max": gps_max,
            "date_start": time_tools.gps2unix(gps_min) if not gps_min_week_time else GPS_WEEK_TIME_ERR_STR,
            "date_end": time_tools.gps2unix(gps_max) if not gps_max_week_time else GPS_WEEK_TIME_ERR_STR,
            "flightline_start": fl_min,
            "flightline_end": fl_max,
            "class_flags": self.get_classification_flags()
        }

        return point_records_summary

    def __crs_info_summary(self) -> dict:

        """
        Summarize the input LAS/LAZ CRS information into a dictionary.

        :return:
        """

        crsinfo = WktCrsInfo(self.get_crs_info())

        return crsinfo.__dict__

    def __vlr_summary(self, evlr=False) -> Union[list[dict], None]:

        """
        Summarize the input LAS/LAZ VLRs or EVLRs into a list of dictionary objects.

        :param evlr: Boolean that controls whether to summarize EVLRs or VLRs.
        :return: list of dictionaries where each list element is a single VLR/EVLR summary.
        """

        records = self.evlrs if evlr else self.vlrs
        if not records:
            return None

        record_summaries = []
        for record in records:

            vlr_num = len(record_summaries) + 1
            vlr_keys = record.__dict__.keys()
            is_copc_info = self.__is_copc_info_vlr(record)
            is_copc_hierarchy = self.__is_copc_hierarchy_vlr(record)
            # laspy names the variable length portion of the VLR differently for each type of VLR.
            # Due to this distasteful decision, we must dynamically assign the variable record data
            # based on each VLR type.
            record_data = ""
            if "record_data" in vlr_keys:
                record_data = record.record_data

            elif is_copc_info:
                record_data = b""  # COPC VLRs are a special case (annoying). Ignore them.

            elif is_copc_hierarchy:
                record_data = record.bytes  # COPC VLRs are a special case (annoying). Ignore them.

            else:
                for key in vlr_keys:
                    if (not key.startswith("_")) and (key not in ["description", "record_id", "user_id"]):
                        record_data = record.__dict__[key]
                        break

            summary = {
                f"vlr{vlr_num}_user_id": record.user_id if not is_copc_hierarchy else None,
                f"vlr{vlr_num}_record_id": record.record_id if not is_copc_hierarchy else None,
                f"vlr{vlr_num}_record_length": sys.getsizeof(record_data) if not is_copc_hierarchy else None,
                f"vlr{vlr_num}_description": record.description,
                f"vlr{vlr_num}_record_data": str(record_data) if not isinstance(record_data, bytes) else None
            }

            record_summaries.append(summary)

        return record_summaries

    def get_guid_asc(self) -> str:

        try:
            guid = uuid.UUID(self.get_guid_hex())
            guid_asc = guid.bytes.decode(UTF_8).replace("\x00", "")
        except UnicodeDecodeError:
            guid_asc = UNICODE_DECODE_ERROR

        return guid_asc

    def __summary_to_json(self, outdir, summary):

        """
        Private method to encapsulate writing summary dict to json file.

        :param outdir: Out directory.
        :param summary: Summary dictionary.
        """

        os.makedirs(outdir, exist_ok=True)
        file_no_ext = os.path.splitext(self.file_basename)[0]
        out_json = os.path.join(outdir, file_no_ext + ".json")
        if not os.path.exists(out_json):
            with open(out_json, "w") as outfile:
                json.dump(summary, outfile, indent=4)

    @staticmethod
    def __format_creation_date(pub_hdr: laspy.LasHeader) -> str:

        """
        Format the creation date attribute for presentation.

        :param pub_hdr: Public header block attribute.
        :return: Date as string.
        """

        creation_date_fmt = ""
        creation_date = pub_hdr.creation_date
        if creation_date:
            creation_month = f"{creation_date.month}" if len(str(creation_date.month)) > 1 else f"0{creation_date.month}"
            creation_day = f"{creation_date.day}" if len(str(creation_date.day)) > 1 else f"0{creation_date.day}"
            creation_date_fmt = f"{creation_date.year}-{creation_month}-{creation_day}"
        return creation_date_fmt

    @staticmethod
    def __is_gps_week_time(gps_time: float) -> bool:

        """
        Check if gps time is gps week time.

        :param gps_time: GPS time (float value)
        :return: True or False
        """

        gpst_no_dec = str(gps_time).split(".")[0]
        return len(gpst_no_dec) <= GPS_WEEK_TIME_LENGTH

    @staticmethod
    def __is_lidar_file(file):

        """
        Determine if filename is LAS/LAZ file.
        """

        return file.endswith("las") or file.endswith("laz")

    @staticmethod
    def __swap_guid_chars(fourth: str, fifth: str):

        strings = ["", ""]
        for i, string in enumerate([fourth, fifth]):
            strings[i] += (string[2:4] + string[0:2])
        strings[1] += fifth[4:]

        return strings[0], strings[1]

    @staticmethod
    def __is_copc_vlr(record):
        return isinstance(record, laspy.copc.CopcHierarchyVlr) or isinstance(record, laspy.copc.CopcInfoVlr)

    @staticmethod
    def __is_copc_info_vlr(record):
        return isinstance(record, laspy.copc.CopcInfoVlr)

    @staticmethod
    def __is_copc_hierarchy_vlr(record):
        return isinstance(record, laspy.copc.CopcHierarchyVlr)


class NotLidarFileError(Exception):
    """
    Exception raised for non LAS/LAZ file input.
    """

    def __init__(self, message="File is not a LAS/LAZ file"):
        self.message = message
        super().__init__(self.message)


def main():

    wackadoo_list = glob.glob("/home/jordan/work/geobc/test_data/wackadoo/*.laz")
    wackadoo_list.extend(
        glob.glob("/home/jordan/work/geobc/test_data/wackadoo/*.las")
    )

    # file = "/media/jordan/EMBC_SKUPPA/problem_files/bcts_092l062_4_2_3_xc_31_12_2012.laz"
    # las = Laszy(file)

    report = LaszyReport(wackadoo_list, las_to_json=True, verbose=True)
    report.write("wacky", validate=True, check_logs=True)


if __name__ == "__main__":
    main()
