# n8n ChromaDB VectorStore Implementation Files

This directory contains the files that were created or modified to add a new ChromaDB VectorStore to n8n's service using the VectorStore abstraction with the Langchain node integration.


## Key Files and Components

### Core Abstraction Layer

#### `shared/createVectorStoreNode/`
The main factory function and base implementation for all VectorStore nodes. This provides:
- **Standardized Operations**: `load`, `insert`, `retrieve`, `update`, `retrieve-as-tool`
- **Common UI Patterns**: Consistent parameter structures across vector stores
- **Error Handling**: Unified error management and validation
- **Embedding Integration**: Seamless integration with Langchain embeddings
- **Batch Processing**: Efficient handling of large document sets

#### `shared/createVectorStoreNode/methods/`
Common methods used across VectorStore implementations:
- **Collection/Index Search**: Dynamic loading of available collections
- **Validation Helpers**: Parameter and data validation utilities
- **Connection Management**: Standardized connection handling patterns

#### `shared/descriptions.ts`
Reusable UI component definitions for different vector stores:
- **Resource Locator Components** (RLC): Dynamic dropdowns for collections/indexes
- **Standard Parameters**: Common fields like collection names, metadata, filters
- **Search Methods**: Integration points for dynamic data loading

## Integration Points

### Langchain Integration
- **Embeddings**: Compatible with all Langchain embedding providers
- **Vector Stores**: Extends Langchain's vector store interfaces
- **Document Processing**: Leverages Langchain's document handling

### n8n Workflow Integration
- **Credential Management**: Secure storage and retrieval of API keys
- **Dynamic UI**: Resource locators that populate from live data
- **Error Handling**: n8n-specific error types and user feedback
- **Tool Integration**: Can be used as tools in AI agent workflows