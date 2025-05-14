from weasyprint import HTML  # ✅ HTML to PDF 변환용
from typing import TypedDict
import markdown2  # 마크다운 → HTML 변환

# ✅ 보고서 저장 함수 (WeasyPrint 기반)
def save_report_to_pdf(markdown_text: str, filename: str = "startup_report.pdf"):
    html_content = markdown2.markdown(markdown_text)

    styled_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Nanum Gothic', sans-serif;
                line-height: 1.6;
                margin: 2em;
            }}
            h1 {{ font-size: 22pt; font-weight: bold; margin-top: 30px; }}
            h2 {{ font-size: 18pt; font-weight: bold; margin-top: 20px; }}
            ul {{ margin-left: 1.5em; }}
            li {{ margin-bottom: 6px; }}
            p {{ margin: 6px 0; font-size: 12pt; }}
            strong {{ font-weight: bold; }}
        </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    HTML(string=styled_html).write_pdf(filename)
    print(f"📄 PDF 저장 완료: {filename}")
