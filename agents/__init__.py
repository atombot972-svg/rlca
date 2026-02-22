from agents.tql import TQLAgent
from agents.tql_ablation import TQLAblationAgent
from agents.q_transformer import QTransformerAgent
try:
    from agents.pac_fql_actor import PACFQLActorAgent
except ModuleNotFoundError:
    PACFQLActorAgent = None


agents = dict(
    tql=TQLAgent,
    tql_ablation=TQLAblationAgent,
    q_transformer=QTransformerAgent,
)
if PACFQLActorAgent is not None:
    agents['pac_fql_actor'] = PACFQLActorAgent
