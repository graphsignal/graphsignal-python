import subprocess

def main():
    subprocess.run(["bash", "tools/update-client.sh"], check=True)

if __name__ == "__main__":
    main()