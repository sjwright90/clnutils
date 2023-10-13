from pandas import read_csv, read_excel, ExcelFile
from pathlib import Path


def load_file_pandas(path, sheet_name=""):
    """
    Imports data from a csv file and returns a pandas dataframe
    """
    file_ext = Path(path).suffix

    if len(sheet_name) == 0:
        xl = ExcelFile(path)
        sheet_name = input(
            f"Please select a sheet from the following: {xl.sheet_names}"
        )

    match file_ext:
        case ".csv":
            df = read_csv(path)
        case ".xlsx":
            df = read_excel(path, sheet_name=sheet_name)
        case _:
            print("File extension not supported")
            return None

    return df


def import_data():
    wdir_raw = Path(
        input("Enter the path to the raw data folder, or entire filepath: ")
        .strip('"')
        .strip("'")
    )

    if wdir_raw.is_file():
        file_name = wdir_raw.name
        wdir_raw = wdir_raw.parent
    else:
        file_name = input(
            "Enter the name of the file to be loaded, with the extension: "
        )

    tab_name = input(
        "Enter the name of the tab to be loaded, if applicable, press enter to skip: "
    )

    df = load_file_pandas(wdir_raw / file_name, tab_name)

    return df
