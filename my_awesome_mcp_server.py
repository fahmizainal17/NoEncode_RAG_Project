# my_awesome_mcp_server.py
import sys, json

def main():
    for line in sys.stdin:
        req = json.loads(line)
        query = req["params"].get("query","")
        # pretend we always return one node whose text is "Echo: {query}"
        result = {"result": [f"Echo: {query}"]}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
