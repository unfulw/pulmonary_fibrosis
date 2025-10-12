import os
import json
import argparse
import pathlib
from typing import Dict, Any, Set, List, Tuple

import pandas as pd
import pydicom
from pydicom.datadict import dictionary_VR

def is_dicom(path: pathlib.Path) -> bool:
    return path.is_file() and (path.suffix.lower() == ".dcm" or path.suffix == "")

def tag_to_str(tag: Tuple[int, int]) -> str:
    return f"({tag[0]:04X},{tag[1]:04X})"

def value_to_scalar(val: Any):
    if isinstance(val, (list, tuple)):
        return "|".join(map(str, val))
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8", errors="strict")
        except Exception:
            return val.hex()
    return val

def element_to_repr(elem: pydicom.dataelem.DataElement):
    from pydicom.sequence import Sequence
    if isinstance(elem.value, Sequence):
        seq_list = []
        for item in elem.value:
            item_dict = {}
            for it_elem in item.iterall():
                if it_elem.tag == (0x7FE0, 0x0010):  #tag for pixel data
                    continue
                key = tag_to_str((it_elem.tag.group, it_elem.tag.element))
                if isinstance(it_elem.value, Sequence):
                    item_dict[key] = "<nested-sequence>"
                else:
                    item_dict[key] = value_to_scalar(it_elem.value)
            seq_list.append(item_dict)
        return json.dumps(seq_list, ensure_ascii = False)
    else:
        return value_to_scalar(elem.value)
    
def scan_folder(
        dicom_dir: pathlib.Path,
        stop_before_pixels: bool,
) -> Set[str]:
    all_cols: Set[str] = set()
    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            fp = pathlib.Path(root) / fn
            if not is_dicom(fp):
                continue
            try:
                ds = pydicom.dcmread(
                    str(fp), stop_before_pixels = stop_before_pixels, force=True
                )
            except Exception:
                continue

            for elem in ds.iterall():
                if elem.tag == (0x7EF0, 0x0010):  #tag for pixel data
                    continue
                all_cols.add(tag_to_str((elem.tag.group, elem.tag.element)))
    return all_cols

def rows_for_folder(
    dicom_dir: pathlib.Path,
    cols: List[str],
    stop_before_pixels: bool,
) -> List[Dict[str, Any]]:
    
    rows: List[Dict[str, Any]] = []
    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            fp = pathlib.Path(root) / fn
            if not is_dicom(fp):
                continue
            rec: Dict[str, Any] = {"_file": str(fp)}
            try:
                ds = pydicom.dcmread(
                    str(fp), stop_before_pixels=stop_before_pixels, force=True
                )
                for c in cols:
                    rec[c] = None
                for elem in ds.iterall():
                    if elem.tag == (0x7EF0, 0x0010):
                        continue
                    key = tag_to_str((elem.tag.group, elem.tag.element))
                    if key in rec:
                        rec[key] = element_to_repr(elem)
            except Exception as e:
                rec["_error"] = f"read_error:{e}"
            rows.append(rec)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out", default="manifests/manifest.csv")
    ap.add_argument("--include-private", action="store_true")
    ap.add_argument("--no-stop-before-pixels", action="store_true")
    args = ap.parse_args()

    dicom_dir = pathlib.Path(args.dicom_dir)
    dicom_dir = dicom_dir.resolve()
    out_path = pathlib.Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop_before_pixels = not args.no_stop_before_pixels

    cols = sorted(
        scan_folder(dicom_dir, stop_before_pixels=stop_before_pixels)
    )

    cols = ["_file"] + cols + ["_error"]

    rows = rows_for_folder(dicom_dir, cols=cols[1:-1], stop_before_pixels=stop_before_pixels)

    df = pd.DataFrame(rows, columns=cols)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows and {len(cols)-2} tag columns.")

if __name__ == "__main__":
    main()