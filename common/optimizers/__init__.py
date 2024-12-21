def get_extra_optimizer(name):
    if name.lower() == "soap":
        from .soap import SOAP
        return SOAP
    if name.lower() == "heavyball":
        from heavyball import PalmForEachSoap
        return PalmForEachSoap
    if name.lower() == "adopt":
        from .adopt import ADOPT
        return ADOPT
    if name.lower() == "heavyball2":
        from heavyball import SFPaLMForeachSOAP
        return SFPaLMForeachSOAP
    if name.lower() == "mars":
        from .mars import MARS
        return MARS
    if name.lower() == "muon":
        from .muon import Muon
        return Muon
    if name.lower() == "kron":
        from .kron import Kron
        return Kron