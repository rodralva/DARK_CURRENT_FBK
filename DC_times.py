import os
import pandas as pd
import glob

def get_duration(File_list):
    """
    Calculates the total duration between the first and last trigger time in a list of files.

    Parameters:
    File_list (list): A list of file paths.

    Returns:
    float: The total duration in seconds.
    """
    df = pd.read_csv(File_list[0], skiprows=2, nrows=50)
    df2 = pd.read_csv(File_list[-1], skiprows=2, nrows=50)

    time_first = df["TrigTime"].iloc[0].split(' ')[-1]
    time_last = df2["TrigTime"].iloc[-1].split(' ')[-1]

    diff_time = pd.to_datetime(time_last) - pd.to_datetime(time_first)
    total_duration = diff_time.total_seconds()
    return total_duration;

df = pd.DataFrame(columns=["N","set","ch","OV","duration"])

for N in [225,247,295,310,337]:
    for s in ["SET1","SET2"]:
        for ch in [1,2,3]:
            for OV in [35,45,7]:
                path = "/media/rodrigoa/Andresito/FBK_Preproduccion/"
                le_path = path + str(N) + "/" + s + "/DC/C" + str(ch) + "--OV" + str(OV) + "**"
                file_list = glob.glob(le_path)
                
                print("-----------------------------------")
                print('Reading files in :' + le_path + '...')
                print('Found {} files'.format(len(file_list)))
                if len(file_list) == 0:
                    print("No files found, exiting...")
                    continue;
                print('First 3 files names:')
                print(file_list[:3])
                total_duration = get_duration(file_list)
                print("Total duration: " + str(total_duration) + " s")
                
                if OV > 10:
                    OV /= 10

                new_row = {"N": N, "set": s, "ch": ch, "OV": OV, "duration": total_duration}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


if os.path.exists("DC_durations.csv"):
    os.remove("DC_durations.csv")

df.to_csv("DC_durations.csv", index=False)
