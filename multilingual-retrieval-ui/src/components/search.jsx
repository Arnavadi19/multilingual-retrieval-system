// src/components/Search.js

import React, { useState } from 'react';
import axios from 'axios';
// Import the CSS Module
import styles from './Search.module.css'; 

// Define the API base URL.
const API_URL = 'http://localhost:8000/api/search'; 

const SearchComponent = () => {
    // State definitions
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [topK, setTopK] = useState(10);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return; 

        setLoading(true);
        setResults([]);
        setError(null);

        try {
            // Axios GET request with parameters
            const response = await axios.get(API_URL, {
                params: {
                    query: query.trim(),
                    top_k: topK
                }
            });
            
            setResults(response.data); 
            
            if (response.data.length === 0) {
                setError("No documents found for this query.");
            }

        } catch (err) {
            console.error("Search API Error:", err);
            
            let errorMessage = "Could not connect to the retrieval service. Please check your API.";
            
            if (err.response) {
                // Server responded with a status code outside the 2xx range
                errorMessage = `Error: ${err.response.status} - ${err.response.data.detail || "Server error occurred."}`;
            } else if (err.request) {
                // Request was made but no response was received
                errorMessage = "No response received from the API. Is the Python server running?";
            }
            
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    return (
        // Apply the main container class
        <div className={styles.container}> 
            
            {/* Apply the title and intro text classes */}
            <h1 className={styles.title}>🌍 Multilingual Retrieval System</h1>
            <p className={styles.introText}>
                Enter your query in **English** to retrieve documents from Hindi, Bengali, and Telugu corpora.
            </p>

            {/* --- Search Form --- */}
            {/* Apply the search form layout class */}
            <form onSubmit={handleSearch} className={styles.searchForm}>
                
                {/* Apply the input classes */}
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your English search query here..."
                    required
                    className={styles.searchInput}
                />
                <input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                    min="1"
                    max="50"
                    title="Number of results (Top K)"
                    className={styles.searchInput}
                    style={{ width: '80px', flexGrow: 0 }} 
                />
                {/* Apply the button class */}
                <button 
                    type="submit" 
                    disabled={loading || !query.trim()} 
                    className={styles.searchButton}
                >
                    {loading ? 'Searching...' : 'Search'}
                </button>
            </form>

            {/* --- Loading / Error / Results Display --- */}
            
            {loading && <p style={{ color: '#61dafb' }}>🔎 Searching for relevant documents...</p>}
            
            {error && !loading && (
                // Apply the error box class
                <div className={styles.errorBox}>
                    **Error:** {error}
                </div>
            )}

            {results.length > 0 && !loading && (
                <>
                    <h2>Retrieved Documents ({results.length})</h2>
                    {/* Apply the results table class */}
                    <table className={styles.resultsTable}>
                        <thead>
                            <tr>
                                {/* Apply the table header class */}
                                <th className={styles.tableHeader} style={{width: '5%'}}>Rank</th>
                                <th className={styles.tableHeader}>Language</th>
                                <th className={styles.tableHeader}>Score (Relevance)</th>
                                <th className={styles.tableHeader} style={{width: '55%'}}>Document Text</th>
                                <th className={styles.tableHeader}>Document ID</th>
                            </tr>
                        </thead>
                        <tbody>
                            {/* Map over the results array */}
                            {results.map((result) => (
                                <tr key={result.doc_id}>
                                    {/* Apply the table cell classes */}
                                    <td className={styles.tableCell}>{result.rank}</td>
                                    {/* Use multiple classes to style language */}
                                    <td className={`${styles.tableCell} ${styles.languageCell}`}>{result.language}</td>
                                    {/* Use multiple classes to style score */}
                                    <td className={`${styles.tableCell} ${styles.scoreCell}`}>{result.score.toFixed(4)}</td>
                                    <td className={`${styles.tableCell} ${styles.documentTextCell}`}> {/* Assuming the API response has a field named 'document_text' */} {result.text || 'Text not available'}</td>
                                    <td className={styles.tableCell} style={{ textAlign: 'left' }}>{result.doc_id}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </>
            )}
        </div>
    );
};

export default SearchComponent;