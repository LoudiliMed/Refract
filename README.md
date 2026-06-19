<p align="center">
  <img src="assets/logo.png" alt="Refract" width="360">
</p>

# Refract

> Réduit jusqu'à 97% des tokens consommés par vos agents IA quand ils utilisent des outils MCP — sans rien perdre.

---

## Le problème en une phrase

Quand votre agent IA (Claude, Cursor...) se connecte à un outil externe — votre calendrier, vos emails, GitHub — il télécharge **la description complète de tous les outils disponibles**, à chaque fois, même s'il n'en utilise qu'un seul.

C'est comme demander à quelqu'un de lire le manuel entier d'un magasin juste pour acheter du pain.

**Refract corrige ça.** Il se place entre votre agent et le serveur d'outils, et ne laisse passer que ce qui est réellement nécessaire.

---

## Ce que ça change concrètement

| | Sans Refract | Avec Refract |
|---|---|---|
| Outils de fichiers (14 outils) | 1 892 tokens | 236 tokens (**−88%**) |
| Outils Google Calendar (5 outils) | 5 010 tokens | 660 tokens (**−87%**) |
| Pack entreprise — Calendar + Gmail + Drive (12 outils) | 8 649 tokens | 882 tokens (**−90%**) |

Moins de tokens envoyés = factures API plus basses, réponses plus rapides.

**Et rien n'est perdu.** Chaque vérification a confirmé que les outils restent utilisables à 100% après compression — aucune information nécessaire n'est supprimée.

---

## Installation

```bash
pip install refract-mcp
```

C'est tout. Aucune clé API requise, aucun compte à créer.

---

## Comment l'utiliser

### Avec Claude Desktop

Ouvrez votre fichier de configuration Claude Desktop et ajoutez :

```json
{
  "mcpServers": {
    "mon-outil-via-refract": {
      "command": "refract-proxy",
      "args": [
        "--target",
        "npx @modelcontextprotocol/server-filesystem /chemin/vers/dossier",
        "--verbose"
      ]
    }
  }
}
```

Remplacez la ligne `--target` par n'importe quel serveur MCP que vous utilisez déjà. Redémarrez Claude Desktop — c'est tout, Refract travaille en arrière-plan.

### En ligne de commande

```bash
refract-proxy --target "npx @modelcontextprotocol/server-filesystem /tmp" --verbose
```

L'option `--verbose` affiche en direct les économies réalisées :

```
[Refract] Connecté à npx @modelcontextprotocol/server-filesystem /tmp
  14 outils  |  1892 → 236 tokens  (88% de réduction)
```

---

## Comment ça marche, sans jargon

Imaginez une bibliothèque avec 50 livres.

**Sans Refract :** votre agent reçoit un résumé détaillé des 50 livres à chaque question, même si la réponse est dans un seul.

**Avec Refract :** votre agent reçoit d'abord une liste de titres (l'index). Une fois qu'il sait quel livre il lui faut, il ne reçoit que le contenu de ce livre-là.

Techniquement :
- **L'index** (toujours envoyé) : juste les noms des outils et une courte description de chacun.
- **Le détail** (envoyé seulement si nécessaire) : la description complète de l'outil utilisé, avec tout ce qu'il faut pour bien l'utiliser — rien de plus, rien de moins.
- **La vérification** : après chaque compression, Refract vérifie automatiquement que rien d'important n'a été supprimé. Si un doute existe, il envoie la version complète plutôt que de prendre un risque.

Aucune intelligence artificielle n'intervient dans ce processus — c'est entièrement automatique, rapide et prévisible.

---

## Compatible avec

- Claude Desktop
- Cursor
- N'importe quel client qui suit le standard MCP (Model Context Protocol)
- N'importe quel serveur MCP existant — vos outils internes, GitHub, Google Workspace, Slack, etc.

---

## Pour les développeurs

### Utilisation en Python

```python
from refract_proxy import RefractProxy

proxy = RefractProxy(
    target_url="npx @modelcontextprotocol/server-filesystem /tmp",
    verbose=True,
)
await proxy.connect()

# Utiliser les outils compressés directement avec l'API Anthropic
tools = proxy.as_anthropic_tools(use_cache=True)

# Ou lancer comme serveur MCP local (stdio)
await proxy.serve()

# Ou l'exposer en HTTP/SSE
await proxy.serve_http()  # → http://localhost:8080/sse
```

### Mode HTTP/SSE

```bash
refract-proxy --target "https://mon-serveur-mcp.com" --mode http --port 8080
```

### Avec un fichier de schémas local (pour tester)

```bash
refract-proxy --target schemas/mcp_calendar_schemas.json --verbose
```

---

## Caching Anthropic intégré

Refract s'intègre avec le [prompt caching d'Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) : `as_anthropic_tools()` marque automatiquement le catalogue compressé comme cachable, ce qui réduit encore plus la facture sur les requêtes répétées.

Exemple sur 30 jours, 100 requêtes/jour, 5 000 tokens de schémas :

| | Coût |
|---|---|
| Sans Refract, sans cache | 45,00 $ |
| Avec Refract + cache | 1,49 $ |

---

## Licence

MIT — utilisation libre, y compris commerciale.
