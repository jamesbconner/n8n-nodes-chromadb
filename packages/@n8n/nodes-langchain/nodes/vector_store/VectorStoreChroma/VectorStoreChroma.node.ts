import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { Embeddings } from '@langchain/core/embeddings';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import type { ChromaLibArgs } from '@langchain/community/vectorstores/chroma';
import { assertParamIsString, type IDataObject, type INodeProperties } from 'n8n-workflow';

import { createVectorStoreNode } from '../shared/createVectorStoreNode/createVectorStoreNode';
import { chromaCollectionsSearch } from '../shared/createVectorStoreNode/methods/listSearch';
import { chromaCollectionRLC } from '../shared/descriptions';

class ExtendedChromaVectorStore extends Chroma {
	private static defaultFilter: IDataObject = {};

	static async fromExistingCollection(
		embeddings: Embeddings,
		args: ChromaLibArgs,
		defaultFilter: IDataObject = {},
	): Promise<Chroma> {
		ExtendedChromaVectorStore.defaultFilter = defaultFilter;
		return await super.fromExistingCollection(embeddings, args);
	}

	async similaritySearch(query: string, k: number, filter?: IDataObject, callbacks?: Callbacks) {
		const mergedFilter = { ...ExtendedChromaVectorStore.defaultFilter, ...filter };
		return await super.similaritySearch(query, k, mergedFilter, callbacks);
	}
}

const sharedFields: INodeProperties[] = [chromaCollectionRLC];

const sharedOptions: INodeProperties[] = [
	{
		displayName: 'Content Payload Key',
		name: 'contentPayloadKey',
		type: 'string',
		default: 'content',
		description: 'The key to use for the content payload in ChromaDB. Default is "content".',
	},
	{
		displayName: 'Metadata Payload Key',
		name: 'metadataPayloadKey',
		type: 'string',
		default: 'metadata',
		description: 'The key to use for the metadata payload in ChromaDB. Default is "metadata".',
	},
];

const insertFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			{
				displayName: 'Clear Collection',
				name: 'clearCollection',
				type: 'boolean',
				default: false,
				description: 'Whether to clear the collection before inserting new documents',
			},
			{
				displayName: 'Distance Function',
				name: 'distanceFunction',
				type: 'options',
				options: [
					{ name: 'Cosine', value: 'cosine' },
					{ name: 'Euclidean', value: 'euclidean' },
					{ name: 'Manhattan', value: 'manhattan' },
				],
				default: 'cosine',
				description: 'Distance function to use for similarity calculations',
			},
			{
				displayName: 'Collection Metadata',
				name: 'collectionMetadata',
				type: 'json',
				default: '{}',
				description: 'JSON metadata to associate with the collection',
			},
			{
				displayName: 'Batch Size',
				name: 'batchSize',
				type: 'number',
				default: 100,
				description: 'Number of documents to process in each batch for embedding operations',
				typeOptions: {
					minValue: 1,
					maxValue: 1000,
				},
			},
			...sharedOptions,
		],
	},
];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [
			{
				displayName: 'Search Filter',
				name: 'searchFilterJson',
				type: 'json',
				typeOptions: {
					rows: 5,
				},
				default:
					'{\n  "$and": [\n    {\n      "metadata.category": {\n        "$eq": "documentation"\n      }\n    }\n  ]\n}',
				validateType: 'object',
				description:
					'Filter documents using ChromaDB\'s <a href="https://docs.trychroma.com/guides/querying#filtering" target="_blank">filtering syntax</a>',
			},
			{
				displayName: 'Include Metadata',
				name: 'includeMetadata',
				type: 'boolean',
				default: true,
				description: 'Whether to include document metadata in search results',
			},
			{
				displayName: 'Metadata Keys',
				name: 'metadataKeys',
				type: 'string',
				default: '',
				placeholder: 'key1,key2,key3',
				description:
					'Comma-separated list of specific metadata keys to include (leave empty for all)',
				displayOptions: {
					show: {
						includeMetadata: [true],
					},
				},
			},
			...sharedOptions,
		],
	},
];

const updateFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [...sharedOptions],
	},
];

export class VectorStoreChroma extends createVectorStoreNode<ExtendedChromaVectorStore>({
	meta: {
		displayName: 'ChromaDB Vector Store',
		name: 'vectorStoreChroma',
		description: 'Work with your data in a ChromaDB collection',
		icon: 'file:chromadb.svg',
		docsUrl:
			'https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstorechroma/',
		credentials: [
			{
				name: 'chromaApi',
				required: true,
			},
		],
		operationModes: ['load', 'insert', 'retrieve', 'update', 'retrieve-as-tool'],
	},
	methods: { listSearch: { chromaCollectionsSearch } },
	loadFields: retrieveFields,
	insertFields,
	sharedFields,
	retrieveFields,
	updateFields,
	async getVectorStoreClient(context, filter, embeddings, itemIndex) {
		const collection = context.getNodeParameter('chromaCollection', itemIndex, '', {
			extractValue: true,
		}) as string;

		const contentPayloadKey = context.getNodeParameter('options.contentPayloadKey', itemIndex, '');
		assertParamIsString('contentPayloadKey', contentPayloadKey, context.getNode());

		const metadataPayloadKey = context.getNodeParameter(
			'options.metadataPayloadKey',
			itemIndex,
			'',
		);
		assertParamIsString('metadataPayloadKey', metadataPayloadKey, context.getNode());

		const credentials = await context.getCredentials('chromaApi');

		const config: ChromaLibArgs = {
			url: credentials.chromaUrl as string,
			collectionName: collection,
			collectionMetadata: contentPayloadKey !== '' ? { contentKey: contentPayloadKey } : undefined,
			...(credentials.apiKey && {
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: credentials.apiKey as string,
					},
				},
			}),
		};

		return await ExtendedChromaVectorStore.fromExistingCollection(embeddings, config, filter);
	},
	async populateVectorStore(context, embeddings, documents, itemIndex) {
		const collectionName = context.getNodeParameter('chromaCollection', itemIndex, '', {
			extractValue: true,
		}) as string;

		const contentPayloadKey = context.getNodeParameter('options.contentPayloadKey', itemIndex, '');
		assertParamIsString('contentPayloadKey', contentPayloadKey, context.getNode());

		const metadataPayloadKey = context.getNodeParameter(
			'options.metadataPayloadKey',
			itemIndex,
			'',
		);
		assertParamIsString('metadataPayloadKey', metadataPayloadKey, context.getNode());

		const clearCollection = context.getNodeParameter(
			'options.clearCollection',
			itemIndex,
			false,
		) as boolean;
		const distanceFunction = context.getNodeParameter(
			'options.distanceFunction',
			itemIndex,
			'cosine',
		) as string;
		const collectionMetadata = context.getNodeParameter(
			'options.collectionMetadata',
			itemIndex,
			'{}',
		) as string;
		const batchSize = context.getNodeParameter('options.batchSize', itemIndex, 100) as number;

		const credentials = await context.getCredentials('chromaApi');

		let parsedCollectionMetadata = {};
		try {
			parsedCollectionMetadata = JSON.parse(collectionMetadata);
		} catch (error) {
			// Use empty object if JSON parsing fails
		}

		const config: ChromaLibArgs = {
			url: credentials.chromaUrl as string,
			collectionName,
			collectionMetadata: {
				...parsedCollectionMetadata,
				...(contentPayloadKey !== '' && { contentKey: contentPayloadKey }),
				...(distanceFunction !== 'cosine' && { distanceFunction }),
			},
			...(credentials.apiKey && {
				chromaClientParams: {
					auth: {
						provider: 'token',
						credentials: credentials.apiKey as string,
					},
				},
			}),
		};

		// Clear collection if requested
		if (clearCollection) {
			try {
				const { ChromaClient } = await import('chromadb');
				const client = new ChromaClient({
					path: credentials.chromaUrl as string,
					...(credentials.apiKey && {
						auth: {
							provider: 'token',
							credentials: credentials.apiKey as string,
						},
					}),
				});

				// Try to delete the collection if it exists
				try {
					await client.deleteCollection({ name: collectionName });
				} catch (error) {
					// Collection might not exist, which is fine
				}
			} catch (error) {
				// ChromaDB client import or operation failed, continue without clearing
			}
		}

		// Process documents in batches if batch size is specified and less than total documents
		if (batchSize > 0 && batchSize < documents.length) {
			for (let i = 0; i < documents.length; i += batchSize) {
				const batch = documents.slice(i, i + batchSize);
				await Chroma.fromDocuments(batch, embeddings, config);
			}
		} else {
			await Chroma.fromDocuments(documents, embeddings, config);
		}
	},
	async releaseVectorStoreClient(_vectorStore: ExtendedChromaVectorStore) {
		// ChromaDB vector store cleanup
		// In ChromaDB, there's no explicit connection to close as it uses HTTP requests
		// However, we can perform any necessary cleanup here if needed in the future
		// For now, this is a no-op but provides the interface for proper resource management
		return Promise.resolve();
	},
}) {}
