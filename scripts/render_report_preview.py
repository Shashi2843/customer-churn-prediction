from jinja2 import Environment, FileSystemLoader
from pathlib import Path
def tiny_png_data_uri(color: str) -> str:
    # 1x1 PNGs encoded as data URIs are enough to verify the template wiring.
    pixels = {
        'purple': b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9X5pQAAAAASUVORK5CYII=',
        'cyan': b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8AABQMBgE7sHfUAAAAASUVORK5CYII=',
        'slate': b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/5+hHgAF8AJ1nG0TAAAAAElFTkSuQmCC',
    }
    encoded = pixels.get(color, pixels['slate'])
    return f"data:image/png;base64,{encoded.decode('ascii')}"

TEMPLATES = Path(__file__).resolve().parents[1] / 'api' / 'templates'
env = Environment(loader=FileSystemLoader(str(TEMPLATES)))

t = env.get_template('report.html')
html = t.render(
    model_info={'model_name':'TestModel','model_type':'XGBoost','preprocessor_type':'StandardScaler'},
    metrics={'AUC':0.92,'Precision':0.78,'Recall':0.65},
    generation_time='2026-05-27 12:00',
    risk_png=tiny_png_data_uri('purple'),
    drift_png=tiny_png_data_uri('cyan'),
    importance_png=tiny_png_data_uri('slate'),
    kpi_png=tiny_png_data_uri('purple')
)
out = Path(__file__).resolve().parents[1] / 'tmp_report.html'
out.write_text(html, encoding='utf-8')
print('WROTE', out)
