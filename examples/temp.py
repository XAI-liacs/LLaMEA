import importlib.metadata

if __name__ == "__main__":
    for mod in importlib.metadata.files("ioh"):
        printable = f"{mod}"
        if "logger" in printable:
            print(printable)
