# my_awesome_mcp_server.py
import sys, json

def main():
    for line in sys.stdin:
        req = json.loads(line)
        q = req["params"].get("query","")
        # wrap the query in a fake result
        resp = {"result": [f"Echo from MCP: {q}"]}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
