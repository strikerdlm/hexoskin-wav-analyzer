# Installation and Setup Guide

## Setting Up a Virtual Environment

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)
- virtualenv or venv (recommended)

### Installation Steps

#### 1. Clone or download the repository
```bash
git clone https://github.com/username/hexoskin-wav-analyzer.git
cd hexoskin-wav-analyzer
```

#### 2. Create a virtual environment
Using venv (built into Python 3):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Using virtualenv:
```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create and activate virtual environment
virtualenv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

After activating your virtual environment:

```bash
python hexoskin_wav_loader.py
```

## Troubleshooting

### Common Issues

#### Missing tkinter
If you encounter an error about missing tkinter:

- **Windows**: Tkinter is included with standard Python installations. Try reinstalling Python with the "tcl/tk and IDLE" option checked.
- **macOS**: Install with Homebrew: `brew install python-tk`
- **Linux (Ubuntu/Debian)**: Install with apt: `sudo apt-get install python3-tk`

#### Missing dependencies
If you encounter errors about missing packages:
```bash
pip install -r requirements.txt --upgrade
```

#### Display issues
If the application doesn't display correctly:
- Try a different theme by modifying the `_set_theme` method in the code
- Ensure your system supports the fonts used (Segoe UI or similar)

## For Developers

### Adding Custom Icons
To add a custom application icon:
1. Create an .ico file (Windows) or .png file (other platforms)
2. Name it "icon.ico" or "icon.png" and place it in the same directory as the application

### Customizing the Theme
The application attempts to use the best available theme for your platform. You can modify the `_set_theme` method in the `HexoskinWavApp` class to use a specific theme.

Available themes may include:
- 'clam' (cross-platform)
- 'vista' (Windows)
- 'aqua' (macOS)
- 'default'

## Contact

For support or questions, please contact Diego Malpica MD at dlmalpicah@unal.edu.co 