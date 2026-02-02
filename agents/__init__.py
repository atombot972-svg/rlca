from agents.tql import TQLAgent
from agents.tql_ablation import TQLAblationAgent
from agents.q_transformer import QTransformerAgent
from agents.pac_fql_actor import PACFQLActorAgent


agents = dict(
    tql=TQLAgent,
    tql_ablation=TQLAblationAgent,
    q_transformer=QTransformerAgent,
    pac_fql_actor=PACFQLActorAgent,
)
