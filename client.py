import logging
from typing import List

import weaviate
from weaviate.auth import _APIKey
from weaviate.classes.query import Filter, QueryReference, MetadataQuery
from weaviate.collections.collection import Collection

from weaviate_client.schema import Chunk, Document


class WeaviateClient:
    def __init__(
        self,
        host: str,
        port: int,
        grpc_host: str,
        grpc_port: int,
        api_key: str | None = None,
    ) -> None:
        print("Initializing MimirWeaviateClient..")
        print(f"-- http_host={host}")
        print(f"-- http_port={port}")
        print(f"-- grpc_host={grpc_host}")
        print(f"-- grpc_port={grpc_port}")

        # self.client = weaviate.use_async_with_custom(
        self.client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            http_secure=False,
            grpc_secure=False,
            auth_credentials=_APIKey(api_key=api_key) if api_key is not None else None,
        )

        # Connect to DocumentNew and MultiVectorChunk collections
        # NOTE: The original collections were Document and Chunk, but those were created with a
        # Weaviate server version of 1.22.6 and weaviate python client version 4.3, and so they didnt support named vectors.
        # Now, this code is using weavite python client version 4.5.7, and expects to connect to a weaviate server
        # version of 1.24 or higher
        # NOTE: -- UPDATE -- And even newer weaviate instance has been deployed using port 8050,
        # which is of version 1.24 or higher, but the collections are called 'Document' and 'MultiVectorChunk'
        # As we are currently switching frequently between versions, we find the collection names dynamically.

        self.document_coll_name = "Document" if port in [8050, 9051] else "DocumentNew"
        self.chunk_coll_name = "Chunk" if port in [8050, 9051] else "MultiVectorChunk"

        self.documents: Collection | None = None
        self.chunks: Collection | None = None

        # self._check_connection()
        self.connect()
        print("MimirWeaviateClient successfully initialized!")

    def connect(self) -> None:
        self.client.connect()
        self.documents = self.client.collections.get(self.document_coll_name)
        self.chunks = self.client.collections.get(self.chunk_coll_name)
        print(f"Weaviate client ready: {self.client.is_ready()}")

    def close(self) -> None:
        self.client.close()

    def get_collection_by_name(self, name: str) -> Collection:
        return self.client.collections.get(name)

    def _parse_chunk_response(self, chunk_return_obj, include_document: bool = True) -> Chunk | None:
        """Parses a chunk response object from a Weaviate search.

        Args:
            chunk_return_obj (weaviate.collections.classes.internal._Object): Weaviate Response Object
            include_document (bool): Whether to include the `document` property as part of the chunk

        Returns:
            Chunk: Chunk response object parsed into a `Chunk` object
        """

        props = chunk_return_obj.properties
        references = chunk_return_obj.references
        props["uuid"] = str(chunk_return_obj.uuid)
        props["page"] = int(props["page"])
        props["order"] = int(props["order"])
        props["nword"] = (
            int(props["nword"]) if props.get("nword", None) is not None else None
        )  # Apparantly, some chunks miss this attribute

        if include_document:
            if "document" not in references or len(references["document"].objects) == 0:
                logging.warning(f'Chunk: {props["chunk_id"]} missing document property, ignoring.')
                return None

            props["document"] = Document(**references["document"].objects[0].properties)
            props["document"].date_published = props["document"].date_published
            props["document"].date_modified = props["document"].date_modified
            props["document"].expiry_date = props["document"].expiry_date
        else:
            props["document"] = None

        if chunk_return_obj.metadata is not None:
            metadata = chunk_return_obj.metadata.__dict__
            props['similarity_score'] = metadata['certainty']
        
        props['vector'] = chunk_return_obj.vector

        return Chunk(**props)

    def get_chunks_of_document(self, doc_id: str, include_vector: bool = True) -> List[Chunk]:
        """Get all chunks of a document.

        Args:
            doc_id (str): the ID of the document
            include_vector (bool, optional): Whether to retrieve the vector/embedding of each chunk.
                Defaults to True.

        Returns:
            List[Chunk]: List of the document's chunks.

        NOTE: As there is no search resulting in a score in this method,
        the score attributes of the returned chunks (of type `Chunk`) will be None.
        """

        response = self.chunks.query.fetch_objects(
            limit=1000,
            filters=Filter.by_ref("document").by_property("doc_id").equal(doc_id)
            & Filter.by_property('nword').greater_or_equal(0),
            return_references=[
                QueryReference(
                    link_on="document",
                    return_properties=[field for field in Document.get_field_names()],
                )
            ],
            include_vector=include_vector,
        )
        return [chunk for chunk in [self._parse_chunk_response(obj) for obj in response.objects] if chunk is not None]

    def _get_chunks(
        self, vector: List[float] | None, named_vector: str, limit: int | None
    ) -> List[Chunk]:
        """Query Weaviate for chunks using vector similarity search and or filters
 
        Args:
            vector (List[float] | None): Embedding to use for the similarity search
            named_vector (str): Which of the named vectors to search amongst
            filters (Any): Filters to use to filter the results
            limit (int | None): Maximum number of chunks to retrieve.
 
        Returns:
            List[Chunk]: List of chunks returned from the search, with additional search metadata
        """
 
        if vector is not None:
            print(f"Searching chunks using filters + vector similarity (with named_vector={named_vector})")
            res = self.chunks.query.near_vector(
                limit=limit,
                near_vector=vector,
                target_vector=named_vector,
                return_references=QueryReference(
                    link_on='document', return_properties=[field for field in Document.get_field_names()]
                ),
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )
 
        else:
            print("Searching chunks only using filters")
            res = self.chunks.query.fetch_objects(
                limit=limit,
                return_references=[
                    QueryReference(
                        link_on='document', return_properties=[field for field in Document.get_field_names()]
                    )
                ],
            )
 
        return [chunk for chunk in [self._parse_chunk_response(obj) for obj in res.objects] if chunk is not None]
 
    def get(
        self,
        vector: List[float] | None,
        named_vector: str | None,
        limit: int | None = None,
    ) -> List[Chunk]:
        """Get chunks from Weaviate using semantic search
 
        Args:
            doc_filters (DocumentFilters | None): Any doc_filters to apply to the search
            chunk_filters (ChunkFilters | None): Any chunk_filters to apply to the search
            vector (List[float] | None): If provided, the search will retrieve chunks based on semantic similarity to the vector
            named_vector (str): Which of the named vectors to search amongst
            limit (int | None, optional): Maximum number of chunks to retrieve. Defaults to None.
 
        Returns:
            List[Chunk]: Result from the search, ranked list of `Chunk` objects
        """
 
        chunks = self._get_chunks(vector=vector, named_vector=named_vector, limit=limit)
 
        return chunks
 
