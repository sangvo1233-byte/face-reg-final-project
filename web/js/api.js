export async function api(path, options = {}) {
    const response = await fetch(path, options);
    const contentType = response.headers.get('content-type') || '';
    const data = contentType.includes('application/json')
        ? await response.json()
        : await response.text();

    if (!response.ok) {
        const message = typeof data === 'object' && data.detail
            ? data.detail
            : `Request failed (${response.status})`;
        throw new Error(message);
    }

    return data;
}
