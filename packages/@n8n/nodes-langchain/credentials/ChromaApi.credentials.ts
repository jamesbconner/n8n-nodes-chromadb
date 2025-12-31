import type {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class ChromaApi implements ICredentialType {
	name = 'chromaApi';

	displayName = 'ChromaDB API';

	documentationUrl = 'https://docs.trychroma.com/';

	properties: INodeProperties[] = [
		{
			displayName: 'ChromaDB URL',
			name: 'chromaUrl',
			type: 'string',
			required: true,
			default: 'http://localhost:8000',
			description: 'The URL of your ChromaDB instance',
			placeholder: 'http://localhost:8000',
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			required: false,
			default: '',
			description: 'Optional API key for authenticated ChromaDB instances',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: '={{$credentials.apiKey ? "Bearer " + $credentials.apiKey : undefined}}',
			},
		},
	};

	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.chromaUrl}}',
			url: '/api/v2/heartbeat',
		},
	};
}
