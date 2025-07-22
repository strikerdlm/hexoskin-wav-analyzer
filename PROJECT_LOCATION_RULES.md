# PROJECT LOCATION RULES - CRITICAL
## Date: 2025-01-14 00:00:00 UTC

### ⚠️ ABSOLUTE PROJECT LOCATION REQUIREMENT

**ONLY PROJECT FOLDER:**
```
C:\Users\User\OneDrive\FAC\Research\Valquiria\Data
```

**❌ NEVER EXPORT TO:**
```
C:\Users\User\OneDrive\FAC\Research\Python Scripts  ← WRONG!
```

### 📂 REQUIRED DIRECTORY STRUCTURE

```
C:\Users\User\OneDrive\FAC\Research\Valquiria\Data/
├── src/
│   └── hrv_analysis/
│       ├── plots_output/          ← ALL PLOTS GO HERE
│       └── enhanced_hrv_analysis/
│           ├── launch_hrv_analysis.py
│           └── gui/
└── docs/
    └── HRV_ANALYSIS_COMPREHENSIVE_DOCUMENTATION.md
```

### 🔧 IMPLEMENTATION FIXES

#### 1. **Working Directory Lock**
**File:** `launch_hrv_analysis.py`
```python
# CRITICAL: Set correct working directory to prevent plots being exported 
# to wrong location
# Project root should be: C:\Users\User\OneDrive\FAC\Research\Valquiria\Data
project_root = current_dir.parent.parent.parent
os.chdir(str(project_root))

# Create plots output directory in the correct project location
plots_output_dir = project_root / "src" / "hrv_analysis" / "plots_output"
plots_output_dir.mkdir(parents=True, exist_ok=True)
```

#### 2. **Export Path Protection**
**File:** `visualization/interactive_plots.py`
```python
# Ensure plots are exported to correct project directory
if not Path(filename).is_absolute():
    # Relative path - use plots_output directory in project
    plots_dir = Path("src/hrv_analysis/plots_output")
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / filename
else:
    # Absolute path provided
    output_path = Path(filename)
```

### ✅ VERIFICATION REQUIREMENTS

Before any commit or release:

1. **Check Working Directory:** Application must print correct working directory on startup
2. **Verify Export Path:** All HTML plots must be saved to `src/hrv_analysis/plots_output/`
3. **Test Plot Generation:** Generate test plots and confirm location
4. **No Python Scripts Path:** Never allow exports to `Python Scripts` folder

### 🚨 PREVENTION RULES

1. **All paths must be relative to the project root**
2. **Working directory MUST be set in launch script**
3. **Create plots_output directory if it doesn't exist**
4. **Log all export paths for verification**
5. **Never use current working directory from external calls**

### 📋 TESTING CHECKLIST

- [ ] Launch application prints correct working directory
- [ ] Generate Poincaré plot → Check it goes to `plots_output/`
- [ ] Generate PSD plot → Check it goes to `plots_output/`
- [ ] Generate Time Series plot → Check it goes to `plots_output/`
- [ ] No files appear in `Python Scripts` folder
- [ ] All plots accessible from project structure

**This ensures all exports stay within the Valquiria project boundaries and prevents path confusion.** 