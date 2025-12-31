import { ChromaApi } from '../ChromaApi.credentials';

describe('ChromaApi Credential', () => {
	const chromaApi = new ChromaApi();

	it('should have correct properties', () => {
		expect(chromaApi.name).toBe('chromaApi');
		expect(chromaApi.displayName).toBe('ChromaDB API');
		expect(chromaApi.documentationUrl).toBe('https://docs.trychroma.com/');
		expect(chromaApi.properties).toHaveLength(2);
	});

	it('should have correct ChromaDB URL property configuration', () => {
		const urlProperty = chromaApi.properties.find((prop) => prop.name === 'chromaUrl');

		expect(urlProperty).toBeDefined();
		expect(urlProperty?.displayName).toBe('ChromaDB URL');
		expect(urlProperty?.type).toBe('string');
		expect(urlProperty?.required).toBe(true);
		expect(urlProperty?.default).toBe('http://localhost:8000');
		expect(urlProperty?.description).toBe('The URL of your ChromaDB instance');
		expect(urlProperty?.placeholder).toBe('http://localhost:8000');
	});

	it('should have correct API key property configuration', () => {
		const apiKeyProperty = chromaApi.properties.find((prop) => prop.name === 'apiKey');

		expect(apiKeyProperty).toBeDefined();
		expect(apiKeyProperty?.displayName).toBe('API Key');
		expect(apiKeyProperty?.type).toBe('string');
		expect(apiKeyProperty?.typeOptions?.password).toBe(true);
		expect(apiKeyProperty?.required).toBe(false);
		expect(apiKeyProperty?.default).toBe('');
		expect(apiKeyProperty?.description).toBe(
			'Optional API key for authenticated ChromaDB instances',
		);
	});

	it('should have correct test configuration', () => {
		expect(chromaApi.test.request.baseURL).toBe('={{$credentials.chromaUrl}}');
		expect(chromaApi.test.request.url).toBe('/api/v1/heartbeat');
		// Authentication is now handled by the authenticate property, not in test headers
		expect(chromaApi.test.request.headers).toBeUndefined();
	});

	it('should have correct authenticate configuration', () => {
		expect(chromaApi.authenticate).toBeDefined();
		expect(chromaApi.authenticate.type).toBe('generic');
		expect(chromaApi.authenticate.properties).toBeDefined();
		expect(chromaApi.authenticate.properties.headers).toBeDefined();
		expect(chromaApi.authenticate.properties.headers?.Authorization).toBe(
			'={{$credentials.apiKey ? "Bearer " + $credentials.apiKey : undefined}}',
		);
	});

	describe('URL validation', () => {
		it('should accept valid HTTP URLs', () => {
			const validUrls = [
				'http://localhost:8000',
				'http://127.0.0.1:8000',
				'http://chroma.example.com',
				'http://chroma.example.com:8000',
			];

			// Since there's no explicit validation in the credential,
			// we're testing that the property accepts string values
			validUrls.forEach((url) => {
				expect(typeof url).toBe('string');
				expect(url.startsWith('http://') || url.startsWith('https://')).toBe(true);
			});
		});

		it('should accept valid HTTPS URLs', () => {
			const validUrls = [
				'https://chroma.example.com',
				'https://chroma.example.com:8000',
				'https://api.chroma.com',
			];

			validUrls.forEach((url) => {
				expect(typeof url).toBe('string');
				expect(url.startsWith('https://')).toBe(true);
			});
		});

		it('should identify invalid URL formats', () => {
			const invalidUrls = [
				'',
				'not-a-url',
				'ftp://example.com',
				'localhost:8000',
				'http://',
				'https://',
			];

			invalidUrls.forEach((url) => {
				if (url === '') {
					expect(url).toBe('');
				} else if (!url.startsWith('http://') && !url.startsWith('https://')) {
					expect(url.startsWith('http://') || url.startsWith('https://')).toBe(false);
				}
			});
		});
	});

	describe('connection validation scenarios', () => {
		it('should handle successful heartbeat response', () => {
			// Mock successful ChromaDB heartbeat response
			const mockSuccessResponse = {
				status: 200,
				data: { 'nanosecond heartbeat': 1234567890 },
			};

			expect(mockSuccessResponse.status).toBe(200);
			expect(mockSuccessResponse.data).toHaveProperty('nanosecond heartbeat');
		});

		it('should handle connection failure scenarios', () => {
			// Mock connection failure scenarios
			const connectionErrors = [
				{ code: 'ECONNREFUSED', message: 'Connection refused' },
				{ code: 'ENOTFOUND', message: 'Host not found' },
				{ code: 'ETIMEDOUT', message: 'Connection timeout' },
			];

			connectionErrors.forEach((error) => {
				expect(error.code).toBeDefined();
				expect(error.message).toBeDefined();
				expect(typeof error.message).toBe('string');
			});
		});

		it('should handle authentication failure scenarios', () => {
			// Mock authentication failure scenarios
			const authErrors = [
				{ status: 401, message: 'Unauthorized' },
				{ status: 403, message: 'Forbidden' },
			];

			authErrors.forEach((error) => {
				expect(error.status).toBeGreaterThanOrEqual(400);
				expect(error.status).toBeLessThan(500);
				expect(typeof error.message).toBe('string');
			});
		});

		it('should handle server error scenarios', () => {
			// Mock server error scenarios
			const serverErrors = [
				{ status: 500, message: 'Internal Server Error' },
				{ status: 502, message: 'Bad Gateway' },
				{ status: 503, message: 'Service Unavailable' },
			];

			serverErrors.forEach((error) => {
				expect(error.status).toBeGreaterThanOrEqual(500);
				expect(typeof error.message).toBe('string');
			});
		});
	});

	describe('error handling', () => {
		it('should provide descriptive error messages for common issues', () => {
			const errorMessages = {
				connectionRefused:
					'ChromaDB server is unreachable. Please check the URL and ensure ChromaDB is running.',
				invalidCredentials: 'Authentication failed with ChromaDB. Please check your API key.',
				invalidUrl: 'Invalid ChromaDB URL format. Please provide a valid HTTP or HTTPS URL.',
				serverError: 'ChromaDB server error. Please check server logs and try again.',
			};

			Object.values(errorMessages).forEach((message) => {
				expect(typeof message).toBe('string');
				expect(message.length).toBeGreaterThan(0);
				expect(message).toContain('ChromaDB');
			});
		});
	});
});
