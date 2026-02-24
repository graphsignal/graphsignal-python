import subprocess

def main():
    subprocess.run(["bash", "tools/build-protobuf.sh"], check=True)

if __name__ == "__main__":
    main()