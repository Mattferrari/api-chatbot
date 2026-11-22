import json

from pydantic import ValidationError

from app.schemas.chat import ChatSalida, HerramientaUsada
from app.schemas.prestamo import PrestamoEntrada
from app.services.modelo import predecir

NOMBRE_TOOL = "calcular_riesgo_impago"
MAX_VUELTAS = 4

SISTEMA = (
    "Eres un asistente de riesgo de un banco. Cuando tengas todos los datos de una solicitud de "
    "prestamo, llama a la herramienta calcular_riesgo_impago. Si falta algun dato, pidelo. "
    "Nunca inventes valores. Responde en espanol y de forma breve."
)


def leer_argumentos(texto):
    """YA HECHO: convierte el texto JSON de la tool en dict ({} si no es JSON valido)."""
    try:
        datos = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(datos, dict):
        return datos
    return {}


def definir_tool():
    """Definicion de la tool para el LLM. Los parametros salen de PrestamoEntrada (un solo contrato)."""
    return {
        "type": "function",
        "function": {
            "name": NOMBRE_TOOL,
            "description": (
                "Calcula la probabilidad de impago de una solicitud de prestamo y devuelve "
                "la decision (aprobar, revisar o rechazar). Necesita los 6 datos."
            ),
            "parameters": PrestamoEntrada.model_json_schema(),
        },
    }


def ejecutar_tool(nombre, argumentos_json):
    """Ejecuta la tool. Nunca lanza excepciones: devuelve siempre un dict con "ok"."""
    if nombre != NOMBRE_TOOL:
        return {"ok": False, "error": f"herramienta desconocida: {nombre}"}
    try:
        datos = json.loads(argumentos_json)
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "error": "los argumentos no son un JSON valido"}
    if not isinstance(datos, dict):
        return {"ok": False, "error": "los argumentos deben ser un objeto JSON"}
    try:
        entrada = PrestamoEntrada(**datos)
    except ValidationError as error:
        detalles = []
        for e in error.errors():
            campo = ".".join(str(x) for x in e["loc"]) or "entrada"
            detalles.append(f"{campo}: {e['msg']}")
        return {"ok": False, "error": "datos invalidos -> " + "; ".join(detalles)}
    try:
        salida = predecir(entrada)
    except Exception as error:  # el modelo no deberia fallar, pero el LLM no debe ver un 500
        return {"ok": False, "error": f"error al ejecutar el modelo: {error}"}
    return {"ok": True, **salida.model_dump()}


def responder(mensajes, llm):
    """Bucle LLM -> tool -> LLM hasta que el LLM conteste con texto (maximo MAX_VUELTAS)."""
    historial = [{"role": "system", "content": SISTEMA}]
    for m in mensajes:
        historial.append({"role": m.role, "content": m.content})
    herramientas = [definir_tool()]
    herramienta_usada = None

    for _ in range(MAX_VUELTAS):
        mensaje = llm.completar(historial, herramientas)
        llamadas = mensaje.get("tool_calls") or []
        if not llamadas:
            return ChatSalida(respuesta=mensaje.get("content") or "", herramienta=herramienta_usada)

        historial.append(mensaje)
        for llamada in llamadas:
            nombre = llamada["function"]["name"]
            argumentos_json = llamada["function"]["arguments"]
            resultado = ejecutar_tool(nombre, argumentos_json)
            herramienta_usada = HerramientaUsada(
                nombre=nombre, argumentos=leer_argumentos(argumentos_json), resultado=resultado
            )
            historial.append(
                {"role": "tool", "tool_call_id": llamada["id"], "content": json.dumps(resultado)}
            )

    return ChatSalida(
        respuesta="No he podido completar el calculo. Revisa los datos de la solicitud e intentalo de nuevo.",
        herramienta=herramienta_usada,
    )
