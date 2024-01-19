import os

for N in [225,247,295,310,337]:
    for set in ["SET1","SET2"]:
        for ch in [1,2,3]:
            for OV in [35,45,7]:
                print("OV: "+str(OV))
                os.system("python3 SiPM_calibration.py --OV "+str(OV)+" --set "+set +" --ch "+str(ch)+" --N "+str(N))
