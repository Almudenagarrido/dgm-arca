# Fase 0 — Propuesta de tema

Antes de escribir una línea de código, cada equipo me entrega una propuesta de una página.
No es burocracia: es la parte más difícil de la práctica y la que más va a determinar
vuestra nota. Un buen tema bien pensado hace que todo lo demás fluya; uno mal elegido os va a
dar problemas en cada fase.

Si no tenéis tema, en [`temas_ejemplo.md`](temas_ejemplo.md) hay una docena desarrollados con
su tarea verificable, sus herramientas y su corpus, y cómo se corregiría cada uno.

Rellenad esta plantilla, guardadla como `docs/propuesta.md` en vuestro repositorio y
enviadme el enlace. Hasta que os diga que está aprobada, no empecéis la fase 1: os ahorraréis
trabajo tirado. Cuando la apruebe, os devolveré por escrito qué se va a evaluar exactamente en
vuestro caso.

---

## Equipo

Nombres y correo de contacto.

Victoria Guillén - 202207977@alu.comillas.edu

Gloria Vidal-Quadras - 202203923@alu.comillas.edu

Almudena Garrido - 202204789@alu.comillas.edu

## El tema en una frase

Qué hace el agente y para quién. Si no cabe en una frase, todavía no está claro.

- Un agente que analiza la descripción de un sistema de IA y determina, con razonamiento y citando el artículo o anexo correspondiente, si cae bajo una práctica prohibida, es de alto riesgo, riesgo limitado o riesgo mínimo según el AI Act europeo, dirigido a responsables de compliance/legal en empresas que van a desplegar sistemas de IA.

## El usuario y su problema

¿Quién es la persona que usaría esto? ¿Qué hace hoy sin vuestro agente? ¿Qué le cuesta:
tiempo, dinero, errores? Si el usuario sois vosotros mismos, decidlo: es perfectamente válido.

- Un responsable de compliance o legal en una empresa que va a lanzar un producto con IA y necesita saber rápido sus obligaciones bajo el AI Act, sin leerse las 144 páginas del reglamento cada vez ni pagar una consulta a un despacho especializado.

## Diez preguntas o tareas reales

De la más fácil a la más difícil. Escribidlas tal como las diría el usuario, no como las
diría un ingeniero. Esto os obliga a pensar en concreto y a mí me sirve para calibrar.

1. ¿Un chatbot de atención al cliente necesita algo especial del AI Act?
2. Filtramos currículums con IA antes de la primera entrevista, ¿es de alto riesgo?
3. ¿Puedo usar reconocimiento facial en cámaras de mi tienda para analizar clientes?
4. Un sistema que ajusta precios de seguros de salud según el perfil del cliente, ¿qué obligaciones tiene?
5. ¿Es legal detectar las emociones de mis empleados en videollamadas?
6. Nuestro sistema de scoring crediticio solo detecta fraude, ¿sigue siendo alto riesgo?
7. Un ayuntamiento quiere priorizar solicitudes de ayudas sociales con IA, ¿qué nivel de riesgo tiene?
8. ¿Cuándo está permitida la identificación biométrica remota en tiempo real por la policía?
9. Mi sistema hace varias cosas (recomienda productos y modera contenido), ¿se aplican reglas distintas a cada parte?
10. ¿Cuándo un sistema del Anexo III deja de considerarse alto riesgo?

## La tarea verificable (fase 1)

¿Qué tipo de problema de vuestro dominio tiene una respuesta que se puede comprobar
automáticamente, sin que nadie la lea? Un número, una fecha, un código que pasa tests, una
consulta que devuelve un resultado, una etiqueta. Describid:

- El tipo de problema y dos ejemplos con su respuesta.
- Cómo lo verificaríais con código: comparar números, ejecutar tests, comparar conjuntos...
- De dónde saldrían unos cientos de problemas con respuesta: un dataset existente, un
  generador, anotación propia.

Dedicadle tiempo a esta sección. Si no encontráis respuesta, el tema no sirve tal como está
y hay que darle una vuelta. Venid a hablarlo.

- Dado un caso de uso descrito en lenguaje natural, clasificarlo en prohibido / alto_riesgo / riesgo_limitado / riesgo_minimo.

- Ejemplo 1: "Una empresa reproduce señales de audio por debajo del umbral de audición en llamadas de venta para empujar compras, y varios clientes se han quejado de compras que no recordaban haber autorizado." → prohibido (Art. 5.1.a).
- Ejemplo 2: "Un sistema de IA analiza los CVs recibidos y descarta automáticamente candidatos antes de que un reclutador humano los vea." → alto_riesgo (Anexo III, 4.a).
- Verificador: comparación exacta contra la etiqueta calculada por un motor de reglas propio (código Python que traduce el Art. 5 y el Anexo III en condiciones booleanas).
- Origen de los problemas: generador propio que muestrea combinaciones de flags y usa el motor de reglas para etiquetar automáticamente cientos/miles de casos; las descripciones en lenguaje natural se generan por plantillas o con ayuda de un LLM, verificadas por muestreo.

## Las herramientas (fase 2)

Al menos tres, una de cada tipo:

- **Consulta fuera del modelo:** qué API o fuente externa real (con enlace a su documentación).
- **Cálculo o ejecución:** qué se calcula o ejecuta y por qué el modelo no debería hacerlo de cabeza.
- **Acción con efecto observable:** qué hace y cómo se comprueba que lo ha hecho.

- Consulta al texto del Reglamento (EU) 2024/1689 vía EUR-Lex; el motor de reglas como calculadora de clasificación; generador de un informe de evaluación de riesgo en PDF con la justificación legal.

## El corpus (fase 3)

- De dónde salen los documentos y cuántos hay (aproximadamente, en documentos y en páginas).
- En qué formato están.
- Con qué licencia, o por qué podéis usarlos.
- Dos ejemplos de pregunta que solo se pueden responder leyendo el corpus.

- Texto completo del Reglamento (EU) 2024/1689 (144 páginas, ya descargado de EUR-Lex), guías de la Comisión sobre prácticas prohibidas y borrador de guías de clasificación de alto riesgo. Documentos oficiales de la UE. Preguntas que solo se responden con el corpus: "¿qué excepciones existen para la identificación biométrica remota policial?", "¿cuándo un sistema del Anexo III deja de ser alto riesgo por el Art. 6.3?".

## Qué puede salir mal

Los dos o tres riesgos que veis (la API tiene límite de peticiones, el corpus está en
imágenes, el verificador es ambiguo...) y qué haríais en cada caso.

- La clasificación legal real tiene zonas grises (mitigación: generamos casos sintéticos desde nuestro propio motor de reglas, no casos reales disputados); el reglamento tiene una actualización reciente (Digital Omnibus 2026) no incluida en el PDF base; las descripciones generadas por LLM podrían no reflejar bien los flags (mitigación: revisión por muestreo).

## Por qué este tema

Dos o tres líneas honestas. "Porque me interesa" es una razón perfectamente buena.

- Nos interesa la intersección de IA y regulación, tiene aplicación práctica real y combina bien clasificación verificable con RAG sobre documentación legal estructurada.