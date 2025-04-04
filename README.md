# AutoSewerDocument

**Automated Sewer Inspection Reporting System**

AutoSewerDocument is a comprehensive tool designed to automate the process of sewer inspection reporting. By integrating advanced image analysis and document generation techniques, it streamlines the creation of detailed inspection reports, enhancing efficiency and accuracy in infrastructure assessments.

---

## Features

- **Image Analysis**: Utilizes advanced algorithms to analyze sewer inspection images, identifying and categorizing defects.
- **Automated Reporting**: Generates detailed inspection reports based on image analysis, reducing manual effort and potential errors.
- **User-Friendly Interface**: Provides a Windows Forms application for seamless interaction and report management.
- **Batch Processing**: Supports batch analysis of multiple inspection images, facilitating large-scale assessments.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SobhanRjz/AutoSewerDocument.git
   cd AutoSewerDocument
   ```
2. **Install Dependencies: Ensure you have Python installed. Then, install the required Python packages**:
  ```bash
  pip install -r requirements.txt
  ```
3. **Set Up the Application: Run the setup script to configure the application**:
  ```bash
  python setup.py install
  ```

## Usage

### Launch the Application
- Run the Windows Forms executable (`AISewerPipes.exe`) found in the root directory.

### Analyze Images & Generate Reports
- Load sewer inspection images through the application.
- The tool automatically analyzes images and generates detailed inspection reports.

### Batch Processing
For bulk analysis, use the batch script:
```bash
python Analyser_Batch.py --input_folder "path/to/your/images" --output_folder "path/to/save/reports"
```



