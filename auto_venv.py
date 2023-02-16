import os
import sys
import venv


def create_venv(venv_dir):
    if 'VIRTUAL_ENV' not in os.environ:
        try:
            # Create the virtual environment
            venv.create(venv_dir, with_pip=True)
            print("Virtual environment created.")
        except Exception as e:
            print(f"An error occurred while creating the virtual environment: {e}")
    else:
        print(f"The virtual environment is already available. {os.environ['VIRTUAL_ENV']}")


def activate_venv(venv_dir):
    try:
        # Activate the virtual environment
        if sys.platform == 'win32':
            activate_path = os.path.join(venv_dir, 'Scripts', 'activate.bat')
        else:
            activate_path = os.path.join(venv_dir, 'bin', 'activate')
        os.system(f'source {activate_path}')

        print("Virtual environment activated.")
    except Exception as e:
        print(f"An error occurred while activating the virtual environment: {e}")


def install_packages(venv_dir, requirements_file):
    try:
        # Install packages in the virtual environment
        os.system(f'source {os.path.join(venv_dir, "bin/activate")} && pip install -r {requirements_file}')
        print("Packages installed.")
    except Exception as e:
        print(f"An error occurred while installing packages: {e}")


def main(venv_dir, requirements_file):
    if not os.path.exists(venv_dir):
        create_venv(venv_dir)

    activate_venv(venv_dir)
    install_packages(venv_dir, requirements_file)


if __name__ == '__main__':
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    requirements_file = 'requirements.txt'
    main(venv_dir, requirements_file)
