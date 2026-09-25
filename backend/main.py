from fastapi import FastAPI


app = FastAPI(
    title="Pokédex Gen 1 API",
    description="Backend do projeto TDE2 de reconhecimento de Pokémon da Primeira Geração.",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "API da Pokédex funcionando!"
    }