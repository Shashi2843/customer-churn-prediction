from jinja2 import Environment, FileSystemLoader
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / 'api' / 'templates'
env = Environment(loader=FileSystemLoader(str(TEMPLATES)))

t = env.get_template('report.html')
html = t.render(
    model_info={'model_name':'TestModel','model_type':'XGBoost','preprocessor_type':'StandardScaler'},
    metrics={'AUC':0.92,'Precision':0.78,'Recall':0.65},
    generation_time='2026-05-27 12:00',
    risk_png='',
    drift_png='',
    importance_png='',
    kpi_png=''
)
out = Path(__file__).resolve().parents[1] / 'tmp_report.html'
out.write_text(html, encoding='utf-8')
print('WROTE', out)
