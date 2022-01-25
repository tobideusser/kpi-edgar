import os
package_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(package_path)

ALLOWED_RELATIONS = [{'kpi', 'cy'}, {'kpi', 'py'}, {'kpi', 'py1'}, {'kpi', 'increase'}, {'kpi', 'increase_py'},
                     {'kpi', 'decrease'}, {'kpi', 'decrease_py'}, {'kpi', 'davon'}, {'kpi', 'davon_coref'},
                     {'kpi', 'attr'},
                     {'kpi_coref', 'cy'}, {'kpi_coref', 'py'}, {'kpi_coref', 'py1'}, {'kpi_coref', 'increase'},
                     {'kpi_coref', 'increase_py'}, {'kpi_coref', 'decrease'}, {'kpi_coref', 'decrease_py'},
                     {'kpi_coref', 'davon'}, {'kpi_coref', 'davon_coref'}, {'kpi_coref', 'attr'},
                     {'davon', 'davon_cy'}, {'davon', 'davon_py'}, {'davon', 'davon_increase'},
                     {'davon', 'davon_decrease'}, {'davon', 'attr'},
                     {'davon_coref', 'davon_cy'}, {'davon_coref', 'davon_py'}, {'davon_coref', 'davon_increase'},
                     {'davon_coref', 'davon_decrease'}, {'davon_coref', 'attr'}, {'davon_coref', 'kpi_coref'}]

ALLOWED_ENTITIES = ["kpi", "cy", "py", "py1", "increase", "increase_py", "decrease", "decrease_py", "davon",
                    "davon_coref", "attr", "davon_cy", "davon_py", "davon_increase", "kpi_coref",
                    "davon_decrease", "false_positive"]

IS_ENTITY_NUMERIC = {
    "kpi": False,
    "kpi_coref": False,
    "cy": True,
    "py": True,
    "py1": True,
    "increase": True,
    "increase_py": True,
    "decrease": True,
    "decrease_py": True,
    "davon": False,
    "davon_coref": False,
    "attr": False,
    "davon_cy": True,
    "davon_py": True,
    "davon_increase": True,
    "davon_decrease": True,
    "false_positive": None
}

ENTITY_ORDER = {
    "kpi": 0,
    "kpi_coref": 0,
    "cy": 3,
    "py": 3,
    "py1": 3,
    "increase": 3,
    "increase_py": 3,
    "decrease": 3,
    "decrease_py": 3,
    "davon": 1,
    "davon_coref": 1,
    "attr": 2,
    "davon_cy": 4,
    "davon_py": 4,
    "davon_increase": 4,
    "davon_decrease": 4,
    "false_positive": 0
}

ALLOWED_1_TO_N_ENTITIES = ['kpi', 'kpi_coref', 'davon', 'davon_coref']
ALLOWED_N_TO_1_ENTITIES = ['davon', 'davon_coref', 'attr']
