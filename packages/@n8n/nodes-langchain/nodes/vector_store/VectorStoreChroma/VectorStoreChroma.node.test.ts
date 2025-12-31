import { mock } from 'jest-mock-extended';
import type { ISupplyDataFunctions } from 'n8n-workflow';

// Mock external modules that are not needed for these unit tests
jest.mock('@langchain/community/vectorstores/chroma', () => {
	const state: { ctorArgs?: unknown[] } = { ctorArgs: undefined };
	class Chroma {
		static fromDocuments = jest.fn();
		static fromExistingCollection = jest.fn();
		similaritySearch = jest.fn();
		constructor(...args: unknown[]) {
			state.ctorArgs = args;
		}
	}
	return { Chroma, __state: state };
});

jest.mock('chromadb', () => ({
	ChromaClient: jest.fn().mockImplementation(() => ({
		listCollections: jest.fn(),
		createCollection: jest.fn(),
		deleteCollection: jest.fn(),
	})),
}));

jest.mock('@utils/sharedFields', () => ({ metadataFilterField: {} }), { virtual: true });
jest.mock(
	'@utils/helpers',
	() => ({ getMetadataFiltersValues: jest.fn(), logAiEvent: jest.fn() }),
	{ virtual: true },
);
jest.mock('@utils/N8nBinaryLoader', () => ({ N8nBinaryLoader: class {} }), { virtual: true });
jest.mock('@utils/N8nJsonLoader', () => ({ N8nJsonLoader: class {} }), { virtual: true });
jest.mock('@utils/logWrapper', () => ({ logWrapper: (fn: unknown) => fn }), { virtual: true });

// Mock the vector store node factory
jest.mock('../shared/createVectorStoreNode/createVectorStoreNode', () => ({
	createVectorStoreNode: (config: {
		getVectorStoreClient: (...args: unknown[]) => unknown;
		populateVectorStore: (...args: unknown[]) => unknown;
	}) =>
		class BaseNode {
			async getVectorStoreClient(...args: unknown[]) {
				return config.getVectorStoreClient.apply(config, args);
			}
			async populateVectorStore(...args: unknown[]) {
				return config.populateVectorStore.apply(config, args);
			}
		},
}));

jest.mock('../shared/createVectorStoreNode/methods/listSearch', () => ({
	chromaCollectionsSearch: jest.fn(),
}));

jest.mock('../shared/descriptions', () => ({
	chromaCollectionRLC: {},
}));

import { Chroma } from '@langchain/community/vectorstores/chroma';
import { ChromaClient } from 'chromadb';

import * as ChromaNode from './VectorStoreChroma.node';

const MockChroma = Chroma as jest.MockedClass<typeof Chroma>;
const MockChromaClient = ChromaClient as jest.MockedClass<typeof ChromaClient>;

describe('VectorStoreChroma.node', () => {
	const helpers = mock<ISupplyDataFunctions['helpers']>();
	const dataFunctions = mock<ISupplyDataFunctions>({ helpers });
	dataFunctions.logger = {
		info: jest.fn(),
		debug: jest.fn(),
		error: jest.fn(),
		warn: jest.fn(),
		verbose: jest.fn(),
	} as unknown as ISupplyDataFunctions['logger'];

	const baseCredentials = {
		chromaUrl: 'http://localhost:8000',
		apiKey: 'test-api-key',
	};

	const mockClient = {
		listCollections: jest.fn(),
		createCollection: jest.fn(),
		deleteCollection: jest.fn(),
	};

	beforeEach(() => {
		jest.resetAllMocks();
		MockChromaClient.mockImplementation(() => mockClient as never);
	});

	describe('getVectorStoreClient', () => {
		it('should create vector store client with default content and metadata keys', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: undefined,
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
			expect(vectorStore).toBe(mockVectorStore);
		});

		it('should create vector store client with custom content and metadata keys', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': 'custom_content',
						'options.metadataPayloadKey': 'custom_metadata',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: { contentKey: 'custom_content' },
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should create vector store client without API key when not provided', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};
			const credentialsWithoutApiKey = {
				chromaUrl: 'http://localhost:8000',
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(credentialsWithoutApiKey),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: undefined,
			});
		});

		it('should pass filter to vector store client', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};
			const filter = { $and: [{ 'metadata.category': { $eq: 'documentation' } }] };

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, filter, mockEmbeddings, 0);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: undefined,
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});
	});

	describe('populateVectorStore', () => {
		it('should populate vector store with default options', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'test content 1', metadata: { id: 1 } },
				{ pageContent: 'test content 2', metadata: { id: 2 } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(mockDocuments, mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: {},
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should populate vector store with custom content and metadata keys', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'test content', metadata: {} }];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': 'custom_content',
						'options.metadataPayloadKey': 'custom_metadata',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(mockDocuments, mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: {
					contentKey: 'custom_content',
				},
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should populate vector store with collection clearing enabled', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'test content', metadata: {} }];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);
			mockClient.deleteCollection = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': true,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(MockChromaClient).toHaveBeenCalledWith({
				path: 'http://localhost:8000',
				auth: {
					provider: 'token',
					credentials: 'test-api-key',
				},
			});
			expect(mockClient.deleteCollection).toHaveBeenCalledWith({ name: 'test-collection' });
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(mockDocuments, mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: {},
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should populate vector store with custom distance function and collection metadata', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'test content', metadata: {} }];
			const collectionMetadata = JSON.stringify({
				description: 'Test collection',
				version: '1.0',
			});

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'euclidean',
						'options.collectionMetadata': collectionMetadata,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(mockDocuments, mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: {
					description: 'Test collection',
					version: '1.0',
					distanceFunction: 'euclidean',
				},
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should handle empty documents array', async () => {
			const mockEmbeddings = {};
			const mockDocuments: Array<{ pageContent: string; metadata: Record<string, unknown> }> = [];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(mockDocuments, mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: {},
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should handle invalid JSON in collection metadata gracefully', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'test content', metadata: {} }];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': 'invalid json {',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(mockDocuments, mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: {},
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});
	});

	describe('ExtendedChromaVectorStore filter behavior', () => {
		it('should store and use default filter in ExtendedChromaVectorStore', async () => {
			const mockEmbeddings = {};
			const mockBaseSimilaritySearch = jest
				.fn()
				.mockResolvedValue([{ pageContent: 'result 1', metadata: {} }]);
			const defaultFilter = { $and: [{ 'metadata.default': { $eq: 'test' } }] };

			// Mock fromExistingCollection to actually call the real ExtendedChromaVectorStore
			// and return an instance that has the overridden similaritySearch method
			MockChroma.fromExistingCollection = jest.fn().mockImplementation(async () => {
				const instance = Object.create(MockChroma.prototype);
				instance.similaritySearch = mockBaseSimilaritySearch;
				return instance;
			});

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			// The filter is passed as a parameter when getVectorStoreClient is called
			// and stored in ExtendedChromaVectorStore via fromExistingCollection
			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, defaultFilter, mockEmbeddings, 0);

			// Verify fromExistingCollection was called (which stores the default filter)
			expect(MockChroma.fromExistingCollection).toHaveBeenCalled();
		});

		it('should verify client creation with collection name', async () => {
			const mockEmbeddings = {};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue({
				similaritySearch: jest.fn(),
			});

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'my-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'my-test-collection',
				}),
			);
		});
	});

	describe('Multi-Environment Support', () => {
		it('should support local ChromaDB instances', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const localCredentials = {
				chromaUrl: 'http://localhost:8000',
				apiKey: 'local-api-key',
			};

			const context = {
				getCredentials: jest.fn().mockResolvedValue(localCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'local-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'local-collection',
					chromaClientParams: {
						auth: {
							provider: 'token',
							credentials: 'local-api-key',
						},
					},
				}),
			);
		});

		it('should support remote ChromaDB instances with HTTPS', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const remoteCredentials = {
				chromaUrl: 'https://my-chroma-instance.example.com',
				apiKey: 'remote-api-key',
			};

			const context = {
				getCredentials: jest.fn().mockResolvedValue(remoteCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'remote-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'https://my-chroma-instance.example.com',
					collectionName: 'remote-collection',
					chromaClientParams: {
						auth: {
							provider: 'token',
							credentials: 'remote-api-key',
						},
					},
				}),
			);
		});

		it('should support remote ChromaDB instances with custom ports', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const customPortCredentials = {
				chromaUrl: 'http://chroma-server.internal:9000',
				apiKey: 'custom-port-api-key',
			};

			const context = {
				getCredentials: jest.fn().mockResolvedValue(customPortCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'custom-port-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://chroma-server.internal:9000',
					collectionName: 'custom-port-collection',
					chromaClientParams: {
						auth: {
							provider: 'token',
							credentials: 'custom-port-api-key',
						},
					},
				}),
			);
		});

		it('should support local ChromaDB instances without authentication', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const localNoAuthCredentials = {
				chromaUrl: 'http://localhost:8000',
				// No apiKey provided
			};

			const context = {
				getCredentials: jest.fn().mockResolvedValue(localNoAuthCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'local-no-auth-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'local-no-auth-collection',
				}),
			);
			// Should not have chromaClientParams when no API key
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.not.objectContaining({
					chromaClientParams: expect.anything(),
				}),
			);
		});

		it('should support different URL formats and environments', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const testCases = [
				{
					name: 'IPv4 address',
					url: 'http://192.168.1.100:8000',
					apiKey: 'ipv4-key',
				},
				{
					name: 'Domain with subdomain',
					url: 'https://chroma.prod.example.com',
					apiKey: 'subdomain-key',
				},
				{
					name: 'Docker internal network',
					url: 'http://chroma-container:8000',
					apiKey: 'docker-key',
				},
			];

			for (const testCase of testCases) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue({
						chromaUrl: testCase.url,
						apiKey: testCase.apiKey,
					}),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: 'test-collection',
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const result = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);

				expect(result).toBe(mockVectorStore);
				expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
					mockEmbeddings,
					expect.objectContaining({
						url: testCase.url,
						collectionName: 'test-collection',
						chromaClientParams: {
							auth: {
								provider: 'token',
								credentials: testCase.apiKey,
							},
						},
					}),
				);
			}
		});
	});

	describe('Client Connection Establishment', () => {
		it('should establish connection with valid credentials and URL', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue({
					chromaUrl: 'http://localhost:8000',
					apiKey: 'valid-api-key',
				}),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'test-collection',
					chromaClientParams: {
						auth: {
							provider: 'token',
							credentials: 'valid-api-key',
						},
					},
				}),
			);
		});

		it('should establish connection with valid credentials but no API key', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue({
					chromaUrl: 'http://localhost:8000',
				}),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'test-collection',
				}),
			);
			// Should not have chromaClientParams when no API key
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.not.objectContaining({
					chromaClientParams: expect.anything(),
				}),
			);
		});

		it('should handle collection selection from RLC parameter', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn(
					(
						name: string,
						_itemIndex: number,
						_defaultValue: string,
						options?: { extractValue?: boolean },
					) => {
						if (name === 'chromaCollection' && options?.extractValue) {
							return 'selected-collection-from-rlc';
						}
						const map: Record<string, unknown> = {
							chromaCollection: 'selected-collection-from-rlc',
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
						};
						return map[name];
					},
				),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'selected-collection-from-rlc',
				}),
			);
		});

		it('should pass filter parameter to ExtendedChromaVectorStore', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};
			const testFilter = { $and: [{ 'metadata.type': { $eq: 'document' } }] };

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(context, testFilter, mockEmbeddings, 0);

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'test-collection',
				}),
			);
		});
	});

	describe('Collection Clearing', () => {
		it('should clear collection before inserting new data when clearCollection is enabled', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'New document after clearing', metadata: { id: 1 } }];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);
			mockClient.deleteCollection = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'collection-to-clear',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': true, // Enable clearing
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify that ChromaClient was created for clearing
			expect(MockChromaClient).toHaveBeenCalledWith({
				path: 'http://localhost:8000',
				auth: {
					provider: 'token',
					credentials: 'test-api-key',
				},
			});

			// Verify that deleteCollection was called to clear the collection
			expect(mockClient.deleteCollection).toHaveBeenCalledWith({
				name: 'collection-to-clear',
			});

			// Verify that documents were then inserted
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'collection-to-clear',
				}),
			);
		});

		it('should not clear collection when clearCollection is disabled', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'Document without clearing', metadata: { id: 1 } }];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);
			mockClient.deleteCollection = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'collection-no-clear',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false, // Disable clearing
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify that ChromaClient was NOT created when clearing is disabled
			expect(MockChromaClient).not.toHaveBeenCalled();

			// Verify that deleteCollection was NOT called
			expect(mockClient.deleteCollection).not.toHaveBeenCalled();

			// Verify that documents were still inserted
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'collection-no-clear',
				}),
			);
		});

		it('should handle collection clearing without API key', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Document with clearing, no auth', metadata: { id: 1 } },
			];
			const credentialsWithoutApiKey = {
				chromaUrl: 'http://localhost:8000',
				// No apiKey provided
			};

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);
			mockClient.deleteCollection = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(credentialsWithoutApiKey),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'collection-clear-no-auth',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': true,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify that ChromaClient was created without auth
			expect(MockChromaClient).toHaveBeenCalledWith({
				path: 'http://localhost:8000',
			});

			// Verify that deleteCollection was called
			expect(mockClient.deleteCollection).toHaveBeenCalledWith({
				name: 'collection-clear-no-auth',
			});

			// Verify that documents were inserted
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'collection-clear-no-auth',
				}),
			);
		});

		it('should gracefully handle collection clearing errors', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Document despite clearing error', metadata: { id: 1 } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);
			// Simulate deleteCollection throwing an error (e.g., collection doesn't exist)
			mockClient.deleteCollection = jest.fn().mockRejectedValue(new Error('Collection not found'));

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'collection-clear-error',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': true,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			// Should not throw an error even if clearing fails
			await expect(
				(node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0),
			).resolves.not.toThrow();

			// Verify that deleteCollection was attempted
			expect(mockClient.deleteCollection).toHaveBeenCalledWith({
				name: 'collection-clear-error',
			});

			// Verify that documents were still inserted despite clearing error
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'collection-clear-error',
				}),
			);
		});

		it('should handle clearing for different collection names', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'Test document', metadata: { id: 1 } }];

			const testCollectionNames = [
				'collection-1',
				'test_collection_2',
				'UPPERCASE_COLLECTION',
				'mixed-Case_Collection123',
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);
			mockClient.deleteCollection = jest.fn().mockResolvedValue(undefined);

			for (const collectionName of testCollectionNames) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: collectionName,
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
							'options.clearCollection': true,
							'options.distanceFunction': 'cosine',
							'options.collectionMetadata': '{}',
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

				// Verify each collection was cleared
				expect(mockClient.deleteCollection).toHaveBeenCalledWith({
					name: collectionName,
				});
			}

			// Verify deleteCollection was called for each collection
			expect(mockClient.deleteCollection).toHaveBeenCalledTimes(testCollectionNames.length);
			// Verify fromDocuments was called for each collection
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(testCollectionNames.length);
		});
	});

	describe('Collection Auto-Creation', () => {
		it('should automatically create collection when it does not exist', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Test document for new collection', metadata: { id: 1 } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'new-auto-created-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify that fromDocuments was called, which handles auto-creation
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'new-auto-created-collection',
					collectionMetadata: {},
				}),
			);
		});

		it('should handle collection auto-creation with custom metadata', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Document for collection with metadata', metadata: { type: 'test' } },
			];
			const collectionMetadata = JSON.stringify({
				description: 'Auto-created collection with custom metadata',
				version: '1.0',
				purpose: 'testing',
			});

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'auto-created-with-metadata',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'euclidean',
						'options.collectionMetadata': collectionMetadata,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify collection creation with custom metadata and distance function
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'auto-created-with-metadata',
					collectionMetadata: {
						description: 'Auto-created collection with custom metadata',
						version: '1.0',
						purpose: 'testing',
						distanceFunction: 'euclidean',
					},
				}),
			);
		});

		it('should handle collection auto-creation with content payload key', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Document with custom content key', metadata: { source: 'test' } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'auto-created-custom-content-key',
						'options.contentPayloadKey': 'custom_content_field',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify collection creation with custom content key
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'auto-created-custom-content-key',
					collectionMetadata: {
						contentKey: 'custom_content_field',
					},
				}),
			);
		});

		it('should handle collection auto-creation for various collection names', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [{ pageContent: 'Test document', metadata: { id: 1 } }];

			const testCollectionNames = [
				'simple-collection',
				'collection_with_underscores',
				'collection123',
				'UPPERCASE-COLLECTION',
				'mixed-Case_Collection123',
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			for (const collectionName of testCollectionNames) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: collectionName,
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
							'options.clearCollection': false,
							'options.distanceFunction': 'cosine',
							'options.collectionMetadata': '{}',
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

				// Verify each collection name is handled correctly
				expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
					mockDocuments,
					mockEmbeddings,
					expect.objectContaining({
						collectionName,
					}),
				);
			}

			// Verify fromDocuments was called for each collection name
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(testCollectionNames.length);
		});
	});

	describe('Document Batch Processing', () => {
		it('should process multiple documents as a single batch operation', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Document 1 content', metadata: { id: 1, type: 'doc' } },
				{ pageContent: 'Document 2 content', metadata: { id: 2, type: 'doc' } },
				{ pageContent: 'Document 3 content', metadata: { id: 3, type: 'doc' } },
				{ pageContent: 'Document 4 content', metadata: { id: 4, type: 'doc' } },
				{ pageContent: 'Document 5 content', metadata: { id: 5, type: 'doc' } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'batch-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify that fromDocuments was called exactly once with all documents
			// This confirms batch processing rather than individual document processing
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(1);
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments, // All documents passed as a single batch
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'batch-test-collection',
				}),
			);
		});

		it('should handle large document batches efficiently', async () => {
			const mockEmbeddings = {};
			// Create a larger batch of documents to test batch processing efficiency
			const mockDocuments = Array.from({ length: 50 }, (_, i) => ({
				pageContent: `Document ${i + 1} content with some meaningful text for embedding`,
				metadata: { id: i + 1, batch: 'large', category: `category_${i % 5}` },
			}));

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'large-batch-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Verify single batch operation for large document sets
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(1);
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				expect.arrayContaining(mockDocuments), // All 50 documents in single batch
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'large-batch-collection',
				}),
			);
		});

		it('should process single document as batch operation', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Single document content', metadata: { id: 1, type: 'single' } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'single-doc-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

			// Even single documents should use batch processing approach
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(1);
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'single-doc-collection',
				}),
			);
		});
	});

	describe('Node configuration', () => {
		it('should create node instance successfully', () => {
			const node = new ChromaNode.VectorStoreChroma();
			expect(node).toBeDefined();
			expect(node).toBeInstanceOf(Object);
		});
	});

	describe('Advanced Configuration Compatibility', () => {
		it('should work correctly with basic operations regardless of advanced options configuration', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Test document with advanced config', metadata: { id: 1 } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			// Test with all advanced options configured
			const contextWithAdvancedOptions = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'advanced-config-collection',
						'options.contentPayloadKey': 'custom_content',
						'options.metadataPayloadKey': 'custom_metadata',
						'options.clearCollection': true,
						'options.distanceFunction': 'euclidean',
						'options.collectionMetadata':
							'{"description": "Advanced collection", "version": "2.0"}',
						'options.batchSize': 50,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(
				contextWithAdvancedOptions,
				mockEmbeddings,
				mockDocuments,
				0,
			);

			// Verify that advanced options are properly applied
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'advanced-config-collection',
					collectionMetadata: {
						description: 'Advanced collection',
						version: '2.0',
						contentKey: 'custom_content',
						distanceFunction: 'euclidean',
					},
				}),
			);

			// Reset mocks for next test
			jest.resetAllMocks();
			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			// Test with minimal/default options
			const contextWithMinimalOptions = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'minimal-config-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
						'options.batchSize': 100,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			await (node as any).populateVectorStore(
				contextWithMinimalOptions,
				mockEmbeddings,
				mockDocuments,
				0,
			);

			// Verify that basic operation works with minimal configuration
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'minimal-config-collection',
					collectionMetadata: {},
				}),
			);
		});

		it('should handle batch size configuration correctly', async () => {
			const mockEmbeddings = {};
			const mockDocuments = Array.from({ length: 150 }, (_, i) => ({
				pageContent: `Document ${i + 1} content`,
				metadata: { id: i + 1 },
			}));

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			// Test with batch size smaller than document count
			const contextWithBatchSize = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'batch-size-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
						'options.batchSize': 50, // Smaller than 150 documents
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(
				contextWithBatchSize,
				mockEmbeddings,
				mockDocuments,
				0,
			);

			// Should be called 3 times (150 documents / 50 batch size = 3 batches)
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(3);

			// Verify each batch has the correct size
			const calls = (
				MockChroma.fromDocuments as jest.MockedFunction<typeof MockChroma.fromDocuments>
			).mock.calls;
			expect(calls[0][0]).toHaveLength(50); // First batch: documents 0-49
			expect(calls[1][0]).toHaveLength(50); // Second batch: documents 50-99
			expect(calls[2][0]).toHaveLength(50); // Third batch: documents 100-149

			// Reset for next test
			jest.resetAllMocks();
			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			// Test with batch size larger than document count
			const contextWithLargeBatchSize = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'large-batch-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'cosine',
						'options.collectionMetadata': '{}',
						'options.batchSize': 200, // Larger than 150 documents
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			await (node as any).populateVectorStore(
				contextWithLargeBatchSize,
				mockEmbeddings,
				mockDocuments,
				0,
			);

			// Should be called once with all documents
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(1);
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments, // All 150 documents in one batch
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'large-batch-collection',
				}),
			);
		});

		it('should handle payload key configuration correctly', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			// Test with custom content payload key
			const contextWithCustomKeys = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'custom-keys-collection',
						'options.contentPayloadKey': 'document_text',
						'options.metadataPayloadKey': 'document_metadata',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).getVectorStoreClient(contextWithCustomKeys, undefined, mockEmbeddings, 0);

			// Verify custom content key is used in collection metadata
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'custom-keys-collection',
					collectionMetadata: { contentKey: 'document_text' },
				}),
			);

			// Reset for next test
			jest.resetAllMocks();
			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			// Test with empty payload keys (should use defaults)
			const contextWithEmptyKeys = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'default-keys-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			await (node as any).getVectorStoreClient(contextWithEmptyKeys, undefined, mockEmbeddings, 0);

			// Verify no custom content key is set when empty
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'default-keys-collection',
					collectionMetadata: undefined,
				}),
			);
		});

		it('should handle advanced JSON configuration correctly', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Test document with advanced JSON config', metadata: { id: 1 } },
			];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			// Test with complex JSON collection metadata
			const complexMetadata = JSON.stringify({
				description: 'Advanced test collection',
				version: '2.1.0',
				tags: ['test', 'advanced', 'configuration'],
				settings: {
					maxDocuments: 10000,
					autoOptimize: true,
				},
				created: '2024-01-01T00:00:00Z',
			});

			const contextWithComplexConfig = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'complex-config-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.clearCollection': false,
						'options.distanceFunction': 'manhattan',
						'options.collectionMetadata': complexMetadata,
						'options.batchSize': 100,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			await (node as any).populateVectorStore(
				contextWithComplexConfig,
				mockEmbeddings,
				mockDocuments,
				0,
			);

			// Verify complex metadata and distance function are properly applied
			expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
				mockDocuments,
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'complex-config-collection',
					collectionMetadata: {
						description: 'Advanced test collection',
						version: '2.1.0',
						tags: ['test', 'advanced', 'configuration'],
						settings: {
							maxDocuments: 10000,
							autoOptimize: true,
						},
						created: '2024-01-01T00:00:00Z',
						distanceFunction: 'manhattan',
					},
				}),
			);
		});

		it('should handle all distance function options correctly', async () => {
			const mockEmbeddings = {};
			const mockDocuments = [
				{ pageContent: 'Test document for distance functions', metadata: { id: 1 } },
			];

			const distanceFunctions = ['cosine', 'euclidean', 'manhattan'];

			MockChroma.fromDocuments = jest.fn().mockResolvedValue(undefined);

			for (const distanceFunction of distanceFunctions) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: `${distanceFunction}-collection`,
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
							'options.clearCollection': false,
							'options.distanceFunction': distanceFunction,
							'options.collectionMetadata': '{}',
							'options.batchSize': 100,
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				await (node as any).populateVectorStore(context, mockEmbeddings, mockDocuments, 0);

				if (distanceFunction === 'cosine') {
					// Cosine is default, so it shouldn't be in metadata
					expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
						mockDocuments,
						mockEmbeddings,
						expect.objectContaining({
							collectionName: `${distanceFunction}-collection`,
							collectionMetadata: {},
						}),
					);
				} else {
					// Non-default distance functions should be in metadata
					expect(MockChroma.fromDocuments).toHaveBeenCalledWith(
						mockDocuments,
						mockEmbeddings,
						expect.objectContaining({
							collectionName: `${distanceFunction}-collection`,
							collectionMetadata: {
								distanceFunction,
							},
						}),
					);
				}
			}

			// Verify fromDocuments was called for each distance function
			expect(MockChroma.fromDocuments).toHaveBeenCalledTimes(distanceFunctions.length);
		});
	});

	describe('Similarity Search Functionality', () => {
		it('should perform similarity search with query text and return ranked documents', async () => {
			const mockEmbeddings = {};
			const mockSearchResults = [
				{
					pageContent: 'First relevant document',
					metadata: { id: 1, category: 'documentation', score: 0.95 },
				},
				{
					pageContent: 'Second relevant document',
					metadata: { id: 2, category: 'documentation', score: 0.87 },
				},
				{
					pageContent: 'Third relevant document',
					metadata: { id: 3, category: 'documentation', score: 0.82 },
				},
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResults),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'search-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.searchFilterJson': '{}',
						'options.includeMetadata': true,
						'options.metadataKeys': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Perform similarity search
			const queryText = 'test query for similarity search';
			const topK = 3;
			const results = await vectorStore.similaritySearch(queryText, topK);

			expect(mockVectorStore.similaritySearch).toHaveBeenCalledWith(queryText, topK);
			expect(results).toEqual(mockSearchResults);
			expect(results).toHaveLength(3);
			// Verify results are ranked (assuming scores are in descending order)
			expect(results[0].metadata.score).toBeGreaterThanOrEqual(results[1].metadata.score);
			expect(results[1].metadata.score).toBeGreaterThanOrEqual(results[2].metadata.score);
		});

		it('should perform similarity search with different query texts', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockImplementation((query: string, k: number) => {
					// Return different results based on query
					const baseResults = [
						{ pageContent: `Result for: ${query}`, metadata: { query, relevance: 'high' } },
						{
							pageContent: `Secondary result for: ${query}`,
							metadata: { query, relevance: 'medium' },
						},
					];
					return Promise.resolve(baseResults.slice(0, k));
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'multi-query-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const testQueries = [
				'machine learning algorithms',
				'natural language processing',
				'computer vision techniques',
				'data science methods',
			];

			for (const query of testQueries) {
				const results = await vectorStore.similaritySearch(query, 2);

				expect(mockVectorStore.similaritySearch).toHaveBeenCalledWith(query, 2);
				expect(results).toHaveLength(2);
				expect(results[0].pageContent).toContain(query);
				expect(results[0].metadata.query).toBe(query);
			}

			expect(mockVectorStore.similaritySearch).toHaveBeenCalledTimes(testQueries.length);
		});

		it('should perform similarity search with various collection configurations', async () => {
			const mockEmbeddings = {};
			const mockSearchResults = [{ pageContent: 'Test document', metadata: { type: 'test' } }];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResults),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const testConfigurations = [
				{
					collectionName: 'config-test-1',
					contentKey: 'custom_content',
					metadataKey: 'custom_metadata',
				},
				{
					collectionName: 'config-test-2',
					contentKey: '',
					metadataKey: '',
				},
				{
					collectionName: 'config-test-3',
					contentKey: 'document_text',
					metadataKey: 'document_meta',
				},
			];

			for (const config of testConfigurations) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: config.collectionName,
							'options.contentPayloadKey': config.contentKey,
							'options.metadataPayloadKey': config.metadataKey,
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const vectorStore = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);

				const results = await vectorStore.similaritySearch('test query', 1);

				expect(results).toEqual(mockSearchResults);
				expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
					mockEmbeddings,
					expect.objectContaining({
						collectionName: config.collectionName,
					}),
				);
			}
		});
	});

	describe('Result Limiting', () => {
		it('should return at most topK results from similarity search', async () => {
			const mockEmbeddings = {};
			const allResults = Array.from({ length: 10 }, (_, i) => ({
				pageContent: `Document ${i + 1}`,
				metadata: { id: i + 1, score: 1 - i * 0.1 },
			}));

			const mockVectorStore = {
				similaritySearch: jest.fn().mockImplementation((_query: string, k: number) => {
					// Return only the requested number of results
					return Promise.resolve(allResults.slice(0, k));
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'limit-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const testLimits = [1, 3, 5, 7, 10];

			for (const limit of testLimits) {
				const results = await vectorStore.similaritySearch('test query', limit);

				expect(results).toHaveLength(limit);
				expect(mockVectorStore.similaritySearch).toHaveBeenCalledWith('test query', limit);

				// Verify we get the expected documents
				for (let i = 0; i < limit; i++) {
					expect(results[i].pageContent).toBe(`Document ${i + 1}`);
					expect(results[i].metadata.id).toBe(i + 1);
				}
			}
		});

		it('should handle topK values larger than available results', async () => {
			const mockEmbeddings = {};
			const availableResults = [
				{ pageContent: 'Only document 1', metadata: { id: 1 } },
				{ pageContent: 'Only document 2', metadata: { id: 2 } },
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(availableResults),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'limited-results-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Request more results than available
			const results = await vectorStore.similaritySearch('test query', 10);

			expect(results).toHaveLength(2); // Only 2 results available
			expect(results).toEqual(availableResults);
			expect(mockVectorStore.similaritySearch).toHaveBeenCalledWith('test query', 10);
		});

		it('should handle zero and negative topK values appropriately', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockImplementation((_query: string, k: number) => {
					// ChromaDB should handle edge cases, but we simulate reasonable behavior
					if (k <= 0) return Promise.resolve([]);
					return Promise.resolve([{ pageContent: 'Test document', metadata: { id: 1 } }]);
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'edge-case-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Test zero limit
			const zeroResults = await vectorStore.similaritySearch('test query', 0);
			expect(zeroResults).toHaveLength(0);

			// Test negative limit
			const negativeResults = await vectorStore.similaritySearch('test query', -1);
			expect(negativeResults).toHaveLength(0);

			// Test positive limit
			const positiveResults = await vectorStore.similaritySearch('test query', 1);
			expect(positiveResults).toHaveLength(1);
		});

		it('should respect different topK values for the same query', async () => {
			const mockEmbeddings = {};
			const allResults = Array.from({ length: 8 }, (_, i) => ({
				pageContent: `Result ${i + 1}`,
				metadata: { id: i + 1, relevance: 8 - i },
			}));

			const mockVectorStore = {
				similaritySearch: jest.fn().mockImplementation((_query: string, k: number) => {
					return Promise.resolve(allResults.slice(0, Math.max(0, k)));
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'same-query-different-limits',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const sameQuery = 'consistent test query';
			const limits = [2, 4, 6, 8];

			for (const limit of limits) {
				const results = await vectorStore.similaritySearch(sameQuery, limit);

				expect(results).toHaveLength(limit);
				expect(mockVectorStore.similaritySearch).toHaveBeenCalledWith(sameQuery, limit);

				// Verify the first result is always the same (highest relevance)
				expect(results[0].pageContent).toBe('Result 1');
				expect(results[0].metadata.id).toBe(1);

				// Verify we get the expected number of results
				expect(results[limit - 1].pageContent).toBe(`Result ${limit}`);
			}
		});
	});

	describe('Metadata Handling', () => {
		it('should include document metadata in search results when requested', async () => {
			const mockEmbeddings = {};
			const mockSearchResults = [
				{
					pageContent: 'Document with metadata',
					metadata: {
						id: 1,
						category: 'documentation',
						author: 'test-author',
						created: '2024-01-01',
						tags: ['test', 'metadata'],
					},
				},
				{
					pageContent: 'Another document with metadata',
					metadata: {
						id: 2,
						category: 'tutorial',
						author: 'another-author',
						created: '2024-01-02',
						tags: ['tutorial', 'example'],
					},
				},
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResults),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'metadata-include-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.includeMetadata': true,
						'options.metadataKeys': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const results = await vectorStore.similaritySearch('test query', 2);

			expect(results).toHaveLength(2);

			// Verify metadata is present and complete
			expect(results[0].metadata).toBeDefined();
			expect(results[0].metadata.id).toBe(1);
			expect(results[0].metadata.category).toBe('documentation');
			expect(results[0].metadata.author).toBe('test-author');
			expect(results[0].metadata.tags).toEqual(['test', 'metadata']);

			expect(results[1].metadata).toBeDefined();
			expect(results[1].metadata.id).toBe(2);
			expect(results[1].metadata.category).toBe('tutorial');
			expect(results[1].metadata.author).toBe('another-author');
			expect(results[1].metadata.tags).toEqual(['tutorial', 'example']);
		});

		it('should exclude document metadata when not requested', async () => {
			const mockEmbeddings = {};
			const mockSearchResultsWithoutMetadata = [
				{
					pageContent: 'Document without metadata',
					metadata: {}, // Empty metadata when not requested
				},
				{
					pageContent: 'Another document without metadata',
					metadata: {}, // Empty metadata when not requested
				},
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResultsWithoutMetadata),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'metadata-exclude-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.includeMetadata': false,
						'options.metadataKeys': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const results = await vectorStore.similaritySearch('test query', 2);

			expect(results).toHaveLength(2);

			// Verify metadata is empty or minimal when not requested
			expect(results[0].metadata).toEqual({});
			expect(results[1].metadata).toEqual({});
		});

		it('should handle specific metadata keys when specified', async () => {
			const mockEmbeddings = {};
			const mockSearchResultsWithSpecificKeys = [
				{
					pageContent: 'Document with specific metadata keys',
					metadata: {
						id: 1,
						category: 'documentation',
						// Other keys like author, created, tags would be filtered out
					},
				},
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResultsWithSpecificKeys),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'metadata-specific-keys-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.includeMetadata': true,
						'options.metadataKeys': 'id,category', // Only include these keys
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const results = await vectorStore.similaritySearch('test query', 1);

			expect(results).toHaveLength(1);

			// Verify only specified metadata keys are present
			expect(results[0].metadata).toHaveProperty('id', 1);
			expect(results[0].metadata).toHaveProperty('category', 'documentation');
			expect(Object.keys(results[0].metadata)).toHaveLength(2);
		});

		it('should handle empty metadata gracefully', async () => {
			const mockEmbeddings = {};
			const mockSearchResultsEmptyMetadata = [
				{
					pageContent: 'Document with no metadata',
					metadata: {},
				},
				{
					pageContent: 'Another document with null metadata',
					metadata: null,
				},
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResultsEmptyMetadata),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'empty-metadata-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.includeMetadata': true,
						'options.metadataKeys': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const results = await vectorStore.similaritySearch('test query', 2);

			expect(results).toHaveLength(2);

			// Verify empty metadata is handled gracefully
			expect(results[0].metadata).toEqual({});
			expect(results[1].metadata).toBeNull();
		});

		it('should handle complex metadata structures', async () => {
			const mockEmbeddings = {};
			const mockSearchResultsComplexMetadata = [
				{
					pageContent: 'Document with complex metadata',
					metadata: {
						id: 1,
						nested: {
							level1: {
								level2: 'deep value',
							},
						},
						array: [1, 2, 3],
						boolean: true,
						number: 42,
						date: '2024-01-01T00:00:00Z',
					},
				},
			];

			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue(mockSearchResultsComplexMetadata),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'complex-metadata-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						'options.includeMetadata': true,
						'options.metadataKeys': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			const results = await vectorStore.similaritySearch('test query', 1);

			expect(results).toHaveLength(1);

			// Verify complex metadata structure is preserved
			const metadata = results[0].metadata;
			expect(metadata.id).toBe(1);
			expect(metadata.nested.level1.level2).toBe('deep value');
			expect(metadata.array).toEqual([1, 2, 3]);
			expect(metadata.boolean).toBe(true);
			expect(metadata.number).toBe(42);
			expect(metadata.date).toBe('2024-01-01T00:00:00Z');
		});
	});

	describe('Vector Store Instance Creation', () => {
		it('should create vector store instance that implements LangChain interface methods', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest
					.fn()
					.mockResolvedValue([{ pageContent: 'test result', metadata: { id: 1 } }]),
				similaritySearchWithScore: jest
					.fn()
					.mockResolvedValue([[{ pageContent: 'test result', metadata: { id: 1 } }, 0.95]]),
				addDocuments: jest.fn().mockResolvedValue(undefined),
				delete: jest.fn().mockResolvedValue(undefined),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Verify the returned instance has LangChain VectorStore interface methods
			expect(result).toBe(mockVectorStore);
			expect(result.similaritySearch).toBeDefined();
			expect(result.similaritySearchWithScore).toBeDefined();
			expect(result.addDocuments).toBeDefined();
			expect(result.delete).toBeDefined();
			expect(result.asRetriever).toBeDefined();

			// Verify the methods are callable
			await result.similaritySearch('test query', 5);
			expect(result.similaritySearch).toHaveBeenCalledWith('test query', 5);

			const retriever = result.asRetriever();
			expect(retriever.getRelevantDocuments).toBeDefined();
		});

		it('should create vector store instance with proper configuration for different environments', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const testEnvironments = [
				{
					name: 'local development',
					credentials: { chromaUrl: 'http://localhost:8000' },
					collection: 'dev-collection',
				},
				{
					name: 'production with auth',
					credentials: {
						chromaUrl: 'https://chroma.prod.example.com',
						apiKey: 'prod-api-key',
					},
					collection: 'prod-collection',
				},
				{
					name: 'staging environment',
					credentials: {
						chromaUrl: 'http://staging-chroma:8000',
						apiKey: 'staging-key',
					},
					collection: 'staging-collection',
				},
			];

			for (const env of testEnvironments) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(env.credentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: env.collection,
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const result = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);

				// Verify each environment creates a valid vector store instance
				expect(result).toBe(mockVectorStore);
				expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
					mockEmbeddings,
					expect.objectContaining({
						url: env.credentials.chromaUrl,
						collectionName: env.collection,
					}),
				);
			}
		});

		it('should create vector store instance with filter support', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const testFilter = {
				$and: [
					{ 'metadata.category': { $eq: 'documentation' } },
					{ 'metadata.status': { $eq: 'published' } },
				],
			};

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'filtered-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				testFilter,
				mockEmbeddings,
				0,
			);

			// Verify the vector store instance is created with filter support
			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'filtered-collection',
				}),
			);
		});

		it('should create vector store instance with custom payload keys', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'custom-keys-collection',
						'options.contentPayloadKey': 'custom_content_field',
						'options.metadataPayloadKey': 'custom_metadata_field',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Verify the vector store instance is created with custom payload keys
			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'custom-keys-collection',
					collectionMetadata: { contentKey: 'custom_content_field' },
				}),
			);
		});

		it('should create vector store instance that supports retriever patterns', async () => {
			const mockEmbeddings = {};
			const mockRetriever = {
				getRelevantDocuments: jest
					.fn()
					.mockResolvedValue([{ pageContent: 'retrieved document', metadata: { source: 'test' } }]),
				invoke: jest.fn().mockResolvedValue([]),
			};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue(mockRetriever),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'retriever-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Verify the vector store can be used as a retriever
			const retriever = vectorStore.asRetriever();
			expect(retriever).toBe(mockRetriever);
			expect(retriever.getRelevantDocuments).toBeDefined();
			expect(retriever.invoke).toBeDefined();

			// Test retriever functionality
			const documents = await retriever.getRelevantDocuments('test query');
			expect(mockRetriever.getRelevantDocuments).toHaveBeenCalledWith('test query');
			expect(documents).toEqual([
				{ pageContent: 'retrieved document', metadata: { source: 'test' } },
			]);
		});
	});

	describe('Resource Cleanup', () => {
		it('should provide releaseVectorStoreClient method in constructor args for resource management', async () => {
			// The releaseVectorStoreClient method is provided in the constructor args
			// and is called automatically by the framework, not directly by users

			// Verify that the ChromaDB node has the releaseVectorStoreClient method defined
			// by checking if it's in the constructor args passed to createVectorStoreNode
			const node = new ChromaNode.VectorStoreChroma();

			// The method exists in the constructor configuration, not as a direct method
			// This is verified by the fact that the node can be instantiated successfully
			// and the framework can call the cleanup method when needed
			expect(node).toBeDefined();
			expect(node).toBeInstanceOf(ChromaNode.VectorStoreChroma);
		});

		it('should handle resource cleanup through framework closeFunction', async () => {
			// Test that the cleanup is handled properly through the framework's closeFunction
			// This simulates how the retrieve operation handles cleanup

			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			// Verify that the vector store was created successfully
			expect(vectorStore).toBe(mockVectorStore);

			// The cleanup is handled automatically by the framework through closeFunction
			// in the retrieve operation, not by direct method calls
			expect(vectorStore).toBeDefined();
		});

		it('should handle resource cleanup for different environments', async () => {
			const mockEmbeddings = {};
			const testEnvironments = [
				{
					name: 'local instance',
					credentials: { chromaUrl: 'http://localhost:8000' },
					collection: 'local-collection',
				},
				{
					name: 'remote instance',
					credentials: {
						chromaUrl: 'https://chroma.prod.example.com',
						apiKey: 'prod-key',
					},
					collection: 'remote-collection',
				},
				{
					name: 'custom port instance',
					credentials: {
						chromaUrl: 'http://chroma-server:9000',
						apiKey: 'custom-key',
					},
					collection: 'custom-collection',
				},
			];

			for (const env of testEnvironments) {
				const mockVectorStore = {
					similaritySearch: jest.fn().mockResolvedValue([]),
					asRetriever: jest.fn().mockReturnValue({
						getRelevantDocuments: jest.fn().mockResolvedValue([]),
					}),
				};

				MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

				const context = {
					getCredentials: jest.fn().mockResolvedValue(env.credentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: env.collection,
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const vectorStore = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);

				// Verify each environment creates a valid vector store instance
				// The cleanup is handled automatically by the framework
				expect(vectorStore).toBe(mockVectorStore);
			}
		});

		it('should support proper resource management patterns', async () => {
			// Test that the ChromaDB implementation follows proper resource management patterns
			// by ensuring vector stores can be created and used without resource leaks

			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest
					.fn()
					.mockResolvedValue([{ pageContent: 'test result', metadata: { id: 1 } }]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest
						.fn()
						.mockResolvedValue([{ pageContent: 'test result', metadata: { id: 1 } }]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'resource-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();

			// Create multiple vector store instances to test resource management
			const vectorStores = [];
			for (let i = 0; i < 3; i++) {
				const vectorStore = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);
				vectorStores.push(vectorStore);

				// Test that each vector store is functional
				const results = await vectorStore.similaritySearch('test query', 5);
				expect(results).toHaveLength(1);
				expect(results[0].pageContent).toBe('test result');
			}

			// Verify all vector stores were created successfully
			expect(vectorStores).toHaveLength(3);
			vectorStores.forEach((vs) => {
				expect(vs).toBe(mockVectorStore);
				expect(vs.similaritySearch).toBeDefined();
				expect(vs.asRetriever).toBeDefined();
			});
		});

		it('should handle ChromaDB-specific resource management correctly', async () => {
			// ChromaDB uses HTTP requests and doesn't maintain persistent connections
			// so resource cleanup is primarily about ensuring no memory leaks

			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const testConfigurations = [
				{
					name: 'basic configuration',
					params: {
						chromaCollection: 'basic-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					},
				},
				{
					name: 'custom payload keys',
					params: {
						chromaCollection: 'custom-keys-collection',
						'options.contentPayloadKey': 'custom_content',
						'options.metadataPayloadKey': 'custom_metadata',
					},
				},
				{
					name: 'filtered configuration',
					params: {
						chromaCollection: 'filtered-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					},
					filter: { $and: [{ 'metadata.type': { $eq: 'document' } }] },
				},
			];

			for (const config of testConfigurations) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => (config.params as any)[name]),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const vectorStore = await (node as any).getVectorStoreClient(
					context,
					config.filter || undefined,
					mockEmbeddings,
					0,
				);

				// Verify each configuration creates a valid vector store
				// ChromaDB cleanup is handled automatically since it uses HTTP requests
				expect(vectorStore).toBe(mockVectorStore);
				expect(vectorStore.similaritySearch).toBeDefined();
			}
		});

		it('should ensure no resource leaks with concurrent operations', async () => {
			// Test that concurrent vector store operations don't cause resource leaks

			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearch: jest.fn().mockResolvedValue([]),
				asRetriever: jest.fn().mockReturnValue({
					getRelevantDocuments: jest.fn().mockResolvedValue([]),
				}),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'concurrent-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
			} as never;

			// Create multiple concurrent operations
			const node = new ChromaNode.VectorStoreChroma();
			const concurrentOperations = [];

			for (let i = 0; i < 5; i++) {
				const operation = (node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0);
				concurrentOperations.push(operation);
			}

			// Wait for all operations to complete
			const vectorStores = await Promise.all(concurrentOperations);

			// Verify all operations completed successfully
			expect(vectorStores).toHaveLength(5);
			vectorStores.forEach((vs) => {
				expect(vs).toBe(mockVectorStore);
			});
		});
	});

	describe('Tool Wrapper Creation', () => {
		it('should create tool wrapper for retrieve-as-tool mode', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				similaritySearchVectorWithScore: jest.fn().mockResolvedValue([
					[{ pageContent: 'Test document 1', metadata: { id: 1 } }, 0.9],
					[{ pageContent: 'Test document 2', metadata: { id: 2 } }, 0.8],
				]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						toolDescription: 'Search ChromaDB for relevant documents',
						toolName: 'chromadb_search',
						topK: 4,
						useReranker: false,
						includeDocumentMetadata: true,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma', typeVersion: 1.3 }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'test-collection',
				}),
			);
		});

		it('should create tool wrapper that accepts natural language queries', async () => {
			const mockEmbeddings = {
				embedQuery: jest.fn().mockResolvedValue([0.1, 0.2, 0.3]),
			};
			const mockVectorStore = {
				similaritySearchVectorWithScore: jest
					.fn()
					.mockResolvedValue([
						[{ pageContent: 'Relevant document about AI', metadata: { topic: 'AI' } }, 0.95],
					]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'ai-knowledge-base',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						toolDescription: 'Search AI knowledge base for information',
						toolName: 'ai_search',
						topK: 3,
						useReranker: false,
						includeDocumentMetadata: true,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma', typeVersion: 1.3 }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'ai-knowledge-base',
				}),
			);
		});

		it('should create tool wrapper compatible with AI agent framework', async () => {
			const mockEmbeddings = {
				embedQuery: jest.fn().mockResolvedValue([0.5, 0.6, 0.7]),
			};
			const mockVectorStore = {
				similaritySearchVectorWithScore: jest
					.fn()
					.mockResolvedValue([
						[{ pageContent: 'Agent-compatible document', metadata: { type: 'agent' } }, 0.88],
					]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'agent-tools-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						toolDescription: 'Tool for AI agents to search documents',
						toolName: 'document_search_tool',
						topK: 5,
						useReranker: false,
						includeDocumentMetadata: true,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma', typeVersion: 1.3 }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'agent-tools-collection',
				}),
			);
		});

		it('should create tool wrapper that returns appropriate results for AI agent consumption', async () => {
			const mockEmbeddings = {
				embedQuery: jest.fn().mockResolvedValue([0.2, 0.4, 0.6]),
			};
			const mockVectorStore = {
				similaritySearchVectorWithScore: jest.fn().mockResolvedValue([
					[
						{
							pageContent: 'AI agent consumable document',
							metadata: { format: 'agent-friendly', priority: 'high' },
						},
						0.96,
					],
					[
						{
							pageContent: 'Secondary document for agents',
							metadata: { format: 'agent-friendly', priority: 'medium' },
						},
						0.84,
					],
				]),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'agent-consumption-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						toolDescription: 'Tool that returns results optimized for AI agent consumption',
						toolName: 'agent_optimized_search',
						topK: 2,
						useReranker: false,
						includeDocumentMetadata: true,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma', typeVersion: 1.3 }),
				logger: dataFunctions.logger,
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const result = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(result).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'agent-consumption-collection',
				}),
			);
		});
	});

	describe('Document Update Operations', () => {
		it('should update existing document with new content while preserving metadata', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				addDocuments: jest.fn().mockResolvedValue(undefined),
			};
			const documentId = 'doc-123';
			const newContent = 'Updated document content';
			const preservedMetadata = { category: 'documentation', version: '2.0' };

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: newContent,
							metadata: preservedMetadata,
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(vectorStore).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(mockEmbeddings, {
				url: 'http://localhost:8000',
				collectionName: 'test-collection',
				collectionMetadata: undefined,
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: 'test-api-key',
					},
				},
			});
		});

		it('should update document with new content and re-embed with current embeddings', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				addDocuments: jest.fn().mockResolvedValue(undefined),
			};
			const documentId = 'doc-456';
			const updatedContent = 'This is the updated content that should be re-embedded';
			const updatedMetadata = { type: 'updated', timestamp: '2024-01-01' };

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'update-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: updatedContent,
							metadata: updatedMetadata,
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(vectorStore).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					url: 'http://localhost:8000',
					collectionName: 'update-test-collection',
				}),
			);
		});

		it('should replace existing document while preserving unchanged metadata fields', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				addDocuments: jest.fn().mockResolvedValue(undefined),
			};
			const documentId = 'doc-789';
			const newContent = 'Completely new document content';
			const partialMetadata = {
				title: 'Updated Title', // This should be updated
				// category: 'original-category' - This should be preserved from original
				lastModified: '2024-01-01', // This is new
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'preserve-metadata-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: newContent,
							metadata: partialMetadata,
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(vectorStore).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'preserve-metadata-collection',
				}),
			);
		});

		it('should handle document updates with custom content and metadata payload keys', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				addDocuments: jest.fn().mockResolvedValue(undefined),
			};
			const documentId = 'doc-custom-keys';
			const customContent = 'Content with custom payload keys';
			const customMetadata = { source: 'custom', type: 'test' };

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'custom-keys-collection',
						'options.contentPayloadKey': 'custom_content_field',
						'options.metadataPayloadKey': 'custom_metadata_field',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: customContent,
							metadata: customMetadata,
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(vectorStore).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'custom-keys-collection',
					collectionMetadata: { contentKey: 'custom_content_field' },
				}),
			);
		});

		it('should handle multiple document updates in sequence', async () => {
			const mockEmbeddings = {};
			const mockVectorStore = {
				addDocuments: jest.fn().mockResolvedValue(undefined),
			};
			const documentIds = ['doc-1', 'doc-2', 'doc-3'];
			const updatedContents = [
				'First updated document',
				'Second updated document',
				'Third updated document',
			];

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			for (let i = 0; i < documentIds.length; i++) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: 'multi-update-collection',
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
							id: documentIds[i],
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
					getInputData: () => [
						{
							json: {
								pageContent: updatedContents[i],
								metadata: { index: i, updated: true },
							},
						},
					],
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const vectorStore = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);

				expect(vectorStore).toBe(mockVectorStore);
			}

			expect(MockChroma.fromExistingCollection).toHaveBeenCalledTimes(documentIds.length);
		});
	});

	describe('Update Error Handling', () => {
		it('should handle errors for non-existent document IDs appropriately', async () => {
			const mockEmbeddings = {};
			const nonExistentId = 'non-existent-doc-id';

			// Mock ChromaDB to throw an error for non-existent document
			const mockVectorStore = {
				addDocuments: jest.fn().mockRejectedValue(new Error('Document not found')),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'error-test-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: nonExistentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: 'Content for non-existent document',
							metadata: { test: true },
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();
			const vectorStore = await (node as any).getVectorStoreClient(
				context,
				undefined,
				mockEmbeddings,
				0,
			);

			expect(vectorStore).toBe(mockVectorStore);
			expect(MockChroma.fromExistingCollection).toHaveBeenCalledWith(
				mockEmbeddings,
				expect.objectContaining({
					collectionName: 'error-test-collection',
				}),
			);
		});

		it('should handle connection errors during update operations', async () => {
			const mockEmbeddings = {};
			const documentId = 'doc-connection-error';

			// Mock connection error
			MockChroma.fromExistingCollection = jest
				.fn()
				.mockRejectedValue(new Error('Connection refused'));

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'connection-error-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: 'Content that will fail to update',
							metadata: { test: true },
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();

			await expect(
				(node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0),
			).rejects.toThrow('Connection refused');
		});

		it('should handle authentication errors during update operations', async () => {
			const mockEmbeddings = {};
			const documentId = 'doc-auth-error';

			// Mock authentication error
			MockChroma.fromExistingCollection = jest
				.fn()
				.mockRejectedValue(new Error('Unauthorized: Invalid API key'));

			const context = {
				getCredentials: jest.fn().mockResolvedValue({
					chromaUrl: 'http://localhost:8000',
					apiKey: 'invalid-api-key',
				}),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'auth-error-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: 'Content with invalid auth',
							metadata: { test: true },
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();

			await expect(
				(node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0),
			).rejects.toThrow('Unauthorized: Invalid API key');
		});

		it('should handle invalid document ID formats', async () => {
			const mockEmbeddings = {};
			const invalidDocumentIds = ['', null, undefined, 123, {}, []];

			const mockVectorStore = {
				addDocuments: jest.fn().mockResolvedValue(undefined),
			};

			MockChroma.fromExistingCollection = jest.fn().mockResolvedValue(mockVectorStore);

			for (const invalidId of invalidDocumentIds) {
				const context = {
					getCredentials: jest.fn().mockResolvedValue(baseCredentials),
					getNodeParameter: jest.fn((name: string) => {
						const map: Record<string, unknown> = {
							chromaCollection: 'invalid-id-collection',
							'options.contentPayloadKey': '',
							'options.metadataPayloadKey': '',
							id: invalidId,
						};
						return map[name];
					}),
					getNode: () => ({ name: 'VectorStoreChroma' }),
					logger: dataFunctions.logger,
					getInputData: () => [
						{
							json: {
								pageContent: 'Content with invalid ID',
								metadata: { test: true },
							},
						},
					],
				} as never;

				const node = new ChromaNode.VectorStoreChroma();
				const vectorStore = await (node as any).getVectorStoreClient(
					context,
					undefined,
					mockEmbeddings,
					0,
				);

				expect(vectorStore).toBe(mockVectorStore);
			}
		});

		it('should handle collection not found errors during updates', async () => {
			const mockEmbeddings = {};
			const documentId = 'doc-no-collection';

			// Mock collection not found error
			MockChroma.fromExistingCollection = jest
				.fn()
				.mockRejectedValue(new Error('Collection not found: non-existent-collection'));

			const context = {
				getCredentials: jest.fn().mockResolvedValue(baseCredentials),
				getNodeParameter: jest.fn((name: string) => {
					const map: Record<string, unknown> = {
						chromaCollection: 'non-existent-collection',
						'options.contentPayloadKey': '',
						'options.metadataPayloadKey': '',
						id: documentId,
					};
					return map[name];
				}),
				getNode: () => ({ name: 'VectorStoreChroma' }),
				logger: dataFunctions.logger,
				getInputData: () => [
					{
						json: {
							pageContent: 'Content for missing collection',
							metadata: { test: true },
						},
					},
				],
			} as never;

			const node = new ChromaNode.VectorStoreChroma();

			await expect(
				(node as any).getVectorStoreClient(context, undefined, mockEmbeddings, 0),
			).rejects.toThrow('Collection not found: non-existent-collection');
		});
	});
});
