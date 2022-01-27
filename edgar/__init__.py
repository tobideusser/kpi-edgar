import os
package_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(package_path)

ALLOWED_RELATIONS = [{'kpi', 'cy'}, {'kpi', 'py'}, {'kpi', 'py1'}, {'kpi', 'increase'}, {'kpi', 'increase_py'},
                     {'kpi', 'decrease'}, {'kpi', 'decrease_py'}, {'kpi', 'thereof'}, {'kpi', 'thereof_coref'},
                     {'kpi', 'attr'},
                     {'kpi_coref', 'cy'}, {'kpi_coref', 'py'}, {'kpi_coref', 'py1'}, {'kpi_coref', 'increase'},
                     {'kpi_coref', 'increase_py'}, {'kpi_coref', 'decrease'}, {'kpi_coref', 'decrease_py'},
                     {'kpi_coref', 'thereof'}, {'kpi_coref', 'thereof_coref'}, {'kpi_coref', 'attr'},
                     # {'thereof', 'thereof_cy'}, {'thereof', 'thereof_py'}, {'thereof', 'thereof_increase'},
                     # {'thereof', 'thereof_decrease'}, {'thereof', 'attr'},
                     {'thereof', 'cy'}, {'thereof', 'py'}, {'thereof', 'increase'}, {'thereof', 'py1'},
                     {'thereof', 'decrease'}, {'thereof', 'attr'},
                     {'thereof_coref', 'thereof_cy'}, {'thereof_coref', 'thereof_py'}, {'thereof_coref', 'thereof_increase'},
                     {'thereof_coref', 'thereof_decrease'}, {'thereof_coref', 'attr'}, {'thereof_coref', 'kpi_coref'}]

ALLOWED_ENTITIES = ["kpi", "cy", "py", "py1", "increase", "increase_py", "decrease", "decrease_py", "thereof",
                    "thereof_coref", "attr", "thereof_cy", "thereof_py", "thereof_increase", "kpi_coref",
                    "thereof_decrease", "false_positive"]

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
    "thereof": False,
    "thereof_coref": False,
    "attr": False,
    "thereof_cy": True,
    "thereof_py": True,
    "thereof_increase": True,
    "thereof_decrease": True,
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
    "thereof": 1,
    "thereof_coref": 1,
    "attr": 2,
    "thereof_cy": 4,
    "thereof_py": 4,
    "thereof_increase": 4,
    "thereof_decrease": 4,
    "false_positive": 0
}

ALLOWED_1_TO_N_ENTITIES = ['kpi', 'kpi_coref', 'thereof', 'thereof_coref']
ALLOWED_N_TO_1_ENTITIES = ['thereof', 'thereof_coref', 'attr']
