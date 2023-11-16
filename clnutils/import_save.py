from pandas import read_csv, read_excel, ExcelFile, DataFrame
from pathlib import Path

# from IPython.display import display
# from ipyfilechooser import FileChooser
# from clrutils import ksm_py_path_str


def load_file_pandas(path) -> DataFrame:
    """
    Imports data from a csv file and returns a pandas dataframe
    """
    file_ext = Path(path).suffix

    match file_ext:
        case ".csv":
            df = read_csv(path)
        case ".xlsx":
            xl = ExcelFile(path)
            sheet_name = input(
                f"Please select a sheet from the following: {xl.sheet_names}"
            )
            df = read_excel(path, sheet_name=sheet_name)
        case _:
            raise ValueError("File type not supported")

    return df


def import_data() -> DataFrame:
    wdir_raw = Path(
        input("Enter the path to the data folder, or entire filepath: ")
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
    # THIS NEEDS WORK, NEED A WAY TO STALL THE SCRIPT UNTIL A FILE IS SELECTED
    # fc = FileChooser(str(ksm_py_path_str))
    # # Set a file filter patern
    # fc.filter_pattern = ["*.csv", "*.xlsx", "*.xls"]
    # display(fc)
    # file_name = fc.selected_filename
    # if file_name is not None:
    #     wdir_raw = Path(fc.selected)

    df = load_file_pandas(wdir_raw / file_name)

    print(f"Loading data from {wdir_raw / file_name}...")

    return df


def save_data(df, counter=0):
    if counter > 5:
        print("Too many attempts. Exiting...")
        return None
    wdir_save = Path(
        input("Enter the path to the folder to save the file, or entire filepath: ")
        .strip('"')
        .strip("'")
    )

    if wdir_save.suffix == "":
        file_name = input(
            "Enter the name of the file to be saved, with the extension: "
        )
        if not file_name.endswith(".csv"):
            file_name += ".csv"
    else:
        file_name = wdir_save.name
        wdir_save = wdir_save.parent

    if (wdir_save / file_name).is_file():
        while True:
            overwrite = input("File already exists. Overwrite? (y/n): ")
            if "y" in overwrite.lower():
                print(f"Overwriting file {wdir_save/file_name}...")
                break
            elif "n" in overwrite.lower():
                save_data(df, counter + 1)
                return None
            else:
                print("Invalid input. Please enter y or n.")
                continue

    if not wdir_save.exists():
        wdir_save.mkdir(parents=True)

    print(f"Saving file to {wdir_save / file_name}...")

    df.to_csv(wdir_save / file_name, index=False, encoding="utf-8-sig")
