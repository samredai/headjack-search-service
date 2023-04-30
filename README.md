# Headjack Search Service (HSS)

This is an open-source implementation of the [Headjack](https://github.com/KnowledgeForge/headjack)
search service specification.

# Getting Started

To get started, clone this repository.

```sh
git clone https://github.com/KnowledgeForge/headjack-search-service
cd headjack-search-service
```

Pull the chroma repo that's included as a submodule.

```sh
git submodule init
git submodule update
```

Start the docker compose environment.

```sh
docker compose up
```

HSS is now available at [http://localhost:16410](http://localhost:16410).
```sh
curl -X 'GET' \
  'http://localhost:16410/query/?text=How%20were%20our%20Q1%20earnings%20this%20year%3F&collection=knowledge&n=1' \
  -H 'accept: application/json'
```

You can find the swagger docs for the API at [http://localhost:16410/docs](http://localhost:16410/docs).

**note**: If you're seeing poor performance of the chroma container on an M1 Mac, make sure you are not using QEMU emulation.
