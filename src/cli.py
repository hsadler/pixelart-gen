from fire import Fire

def hello():
    print("Hello World!")

if __name__ == "__main__":
    Fire({
        "hello": hello
    })
