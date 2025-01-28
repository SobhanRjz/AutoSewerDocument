import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.worksheet.worksheet import Worksheet

@dataclass
class ExcelStyles:
    """Styles for Excel formatting"""
    header_font: Font = Font(bold=True, color="FFFFFF") 
    header_alignment: Alignment = Alignment(horizontal="center", vertical="center")
    header_fill: PatternFill = PatternFill(start_color="16365C", end_color="16365C", fill_type="solid")
    header_border: Border = Border(
        left=Side(style="thin", color="000000"),
        right=Side(style="thin", color="000000"),
        top=Side(style="thin", color="000000"),
        bottom=Side(style="thin", color="000000")
    )
    data_fill_even: PatternFill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    data_fill_odd: PatternFill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    data_alignment: Alignment = Alignment(horizontal="center", vertical="center")
    data_font: Font = Font(color="000000")

class ExcelReporter:
    def __init__(self, excelOutPutName: str):
        self.output_path = os.path.join("output", excelOutPutName)
        self.DefectsCodeDict = {}
        self.workbook = Workbook()
        self.sheet = self.workbook.active
        self.sheet.title = "Condition Details"
        self.styles = ExcelStyles()
        self.headers = [
            "زمان بر روی ویدئو", "مسیر تصویر", "فاصله از نقطه شروع", "عیوب پیوسته", "کد عیوب",
            "نام فارسی عیوب", "عیب در محل اتصال", "جنس", "شدت عیب", "ابعاد یک", "ابعاد دو",
            "درصد", "محل قرارگیری از ساعت", "محل قرارگیری تا ساعت", "امتیاز بهره برداری",
            "امتیاز سازه ای", "ملاحضات"
        ]
        self.data_mapping = {
            "زمان بر روی ویدئو": "timestamp_seconds",
            "کد عیوب": "class",
            "شدت عیب": "confidence"
        }

    def load_json_data(self, path) -> List[Dict]:
        """Load and return JSON data from file"""
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def apply_header_styling(self) -> None:
        """Apply styling to header row"""
        for col_num, header in enumerate(self.headers, 1):
            cell = self.sheet.cell(row=1, column=col_num, value=header)
            cell.font = self.styles.header_font
            cell.alignment = self.styles.header_alignment
            cell.fill = self.styles.header_fill
            cell.border = self.styles.header_border

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def write_data_rows(self, json_data: List[Dict]) -> None:
        """Write and style data rows"""
        for row_num, entry in enumerate(json_data, 2):
            fill = self.styles.data_fill_even if row_num % 2 == 0 else self.styles.data_fill_odd
            for col_num, header in enumerate(self.headers, 1):
                cell = self.sheet.cell(row=row_num, column=col_num)
                key = self.data_mapping.get(header)

                if header:
                    if header == "زمان بر روی ویدئو":
                        value = self.format_timestamp(float(entry.get(key, 0)))
                    elif header == "کد عیوب":
                        raw_value = entry.get(key, "")
                        value = self.DefectsCodeDict.get(raw_value, raw_value)
                    else:
                        value = entry.get(key, "")
                else:
                    value = ""

                cell.value = value
                cell.font = self.styles.data_font
                cell.alignment = self.styles.data_alignment
                cell.border = self.styles.header_border
                cell.fill = fill

    def adjust_column_widths(self) -> None:
        """Adjust column widths based on content"""
        for col in self.sheet.columns:
            max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            self.sheet.column_dimensions[col[0].column_letter].width = max_length + 2
    def createCodeMapData(self, defects_data: List[Dict]) -> None:
        # Extract BasicName-to-Key mapping
        basic_to_key_mapping = {}
        for defect_category in defects_data.values():
            if isinstance(defect_category, dict) and "BasicDefects" in defect_category:
                for key, details in defect_category.items():
                    if isinstance(details, dict) and "BasicName" in details:
                        basic_to_key_mapping[details["BasicName"]] = key
        self.DefectsCodeDict = basic_to_key_mapping

    def generate_report(self) -> None:
        """Generate the Excel report"""
        detectionDataPath = r"C:\Users\sobha\Desktop\detectron2\Code\Auto_Sewer_Document\output\Closed circuit television (CCTV) sewer inspection_detections.json"
        defectCodePath = "DefectsCode.json"

        detectionData = self.load_json_data(detectionDataPath)
        defectCodeData = self.load_json_data(defectCodePath)

        self.apply_header_styling()
        self.createCodeMapData(defectCodeData)
        self.write_data_rows(detectionData)
        self.adjust_column_widths()
        self.workbook.save(self.output_path)
        print(f"Excel file with modern styling created: {self.output_path}")

if __name__ == "__main__":
    reporter = ExcelReporter(
        "modern_styled_condition_details_filled.xlsx"
    )
    reporter.generate_report()
