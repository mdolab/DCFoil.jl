# --- Python ---
"""
@File          :   setup_juliapkg.py
@Date created  :   2025/02/05
@Last modified :   2025/05/22
@Author        :   Galen Ng
@Desc          :   Simple file to parse the Project.toml file and update the PythonCall Julia package manager file
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
import juliapkg

fname = "Project.toml"
targetFile = "./juliapkg.json"
targetFile = None  # Set to None to use default location...I can't seem to get the isolated install to work so this is a workaround
# This will not work if multiple julia environments are needed

# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":

    juliapkg.require_julia("1.11", target=targetFile)

    f = open(fname, "r").read()
    dependencies = f.split("\n\n")[1]
    if "[deps]" in dependencies.split("\n")[0]:
        for line in dependencies.split("\n")[1::]:
            pkgName = line.split('"')[0].split(" ")[0]
            uuid = line.split('"')[1]

            juliapkg.add(pkgName, uuid, target=targetFile)

        juliapkg.resolve()
    else:
        print("WARNING: No dependencies found")
