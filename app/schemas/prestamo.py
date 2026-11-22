from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class PrestamoEntrada(BaseModel):
    """Datos de una solicitud de prestamo (entrada del modelo y argumentos de la tool)."""

    # extra="forbid": si llega un campo que no existe (p. ej. inventado por el LLM) -> error
    model_config = ConfigDict(extra="forbid")

    edad: int = Field(ge=18, le=75, description="Edad del solicitante en anios (18 a 75)")
    ingresos_mensuales: float = Field(gt=0, description="Ingresos netos mensuales en euros")
    importe_prestamo: float = Field(gt=0, description="Importe solicitado en euros")
    plazo_meses: int = Field(ge=6, le=84, description="Plazo de devolucion en meses (6 a 84)")
    antiguedad_laboral_anios: int = Field(
        ge=0, le=50, description="Anios en el empleo actual (0 a 50, como mucho edad - 16)"
    )
    impagos_previos: int = Field(ge=0, le=20, description="Numero de impagos anteriores (0 a 20)")

    @field_validator("antiguedad_laboral_anios")
    @classmethod
    def antiguedad_coherente_con_edad(cls, valor: int, info: ValidationInfo) -> int:
        edad = info.data.get("edad")  # no esta si la edad ya fallo su propia validacion
        if edad is not None and valor > edad - 16:
            raise ValueError(
                f"la antiguedad laboral ({valor}) no puede ser mayor que edad - 16 ({edad - 16})"
            )
        return valor


class PrediccionSalida(BaseModel):
    """Resultado del modelo."""

    probabilidad_impago: float = Field(ge=0, le=1)
    decision: Literal["aprobar", "revisar", "rechazar"]
