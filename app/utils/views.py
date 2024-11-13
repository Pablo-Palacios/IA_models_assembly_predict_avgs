def avgs_(at_bats,hits):
    if at_bats == 0:
        return 0
    resultado = float(hits/at_bats)
    return round(resultado,3)

def total_at_bats(aparciones, bb, hbp, sf):
    ab = aparciones - bb - hbp - sf
    return ab

def resta(a,b):
    return a - b

def multiplicacion(a,b):
    return a * b

def division(a,b):
    return a/b


def calcular_promedio_bateo(apariciones,hits,bb,hbp,sf):
    at_basts = total_at_bats(apariciones,bb,hbp,sf)
    avg = avgs_(hits,at_basts)
    return avg
