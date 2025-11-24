import React, { useState } from 'react';
import axios from 'axios';

// Define the API base URL.
const API_URL = 'http://127.0.0.1:8000/api/search'; 

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
            
            // Handle response data, checking for a potential 'results' field or just the array
            setResults(response.data.results || response.data); 
            
            if ((response.data.results || response.data).length === 0) {
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
                errorMessage = `No response received from the API at ${API_URL}. Is the Python server running?`;
            }
            
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    // Helper function to determine language-specific badge colors
    const getLanguageStyle = (language) => {
        switch (language) {
            case 'hi': // Hindi
                return { backgroundColor: '#ff9900', color: '#1a1a1a' };
            case 'bn': // Bengali
                return { backgroundColor: '#4caf50', color: '#1a1a1a' };
            case 'te': // Telugu
                return { backgroundColor: '#2196f3', color: '#1a1a1a' };
            default:
                return { backgroundColor: '#666', color: '#f0f0f0' };
        }
    };

    return (
        <div className="main-app-wrapper">
            {/* Inline CSS for the dark theme and layout */}
            <style>
                {`
                /* Global body styles for the dark theme */
                .main-app-wrapper {
                    display: flex;
                    justify-content: center;
                    align-items: flex-start;
                    min-height: 100vh;
                    background-color: #1a1a1a;
                    padding-top: 40px;
                    padding-bottom: 40px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                    sans-serif;
                }

                /* --- Main Layout and Typography --- */
                .container {
                    /* Main wrapper styles */
                    width: 90%;
                    max-width: 1200px;
                    background-color: #242424; /* Slightly lighter inner background */
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); /* Deep shadow for depth */
                    border: 1px solid #333;
                }

                .title {
                    color: #61dafb; /* React/Vite accent color */
                    font-weight: 700;
                    margin-bottom: 5px;
                    font-size: 2.5em;
                    text-align: center;
                }

                .introText {
                    color: #aaa;
                    margin-bottom: 25px;
                    text-align: center;
                    font-size: 1.1em;
                }

                /* --- Search Bar Styling --- */
                .searchForm {
                    display: flex;
                    flex-wrap: wrap; /* Allows wrapping on smaller screens */
                    gap: 10px;
                    margin-bottom: 30px;
                    align-items: center;
                }

                .searchInput {
                    flex-grow: 1;
                    padding: 12px 15px;
                    font-size: 1em;
                    border: 1px solid #444;
                    border-radius: 8px;
                    background-color: #333;
                    color: #f0f0f0;
                    transition: border-color 0.3s;
                    min-width: 100px;
                }

                .searchInput:focus {
                    border-color: #61dafb;
                    outline: none;
                }

                .searchButton {
                    padding: 12px 25px;
                    font-size: 1em;
                    font-weight: 700;
                    background-color: #61dafb;
                    color: #1a1a1a;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: background-color 0.3s, transform 0.1s;
                }

                .searchButton:hover {
                    background-color: #4dc2ea;
                }

                .searchButton:active {
                    transform: scale(0.98);
                }

                .searchButton:disabled {
                    background-color: #444;
                    cursor: not-allowed;
                }

                /* --- Results Table Styling --- */
                .resultsContainer {
                    max-height: 75vh; 
                    overflow-y: auto; 
                    border: 1px solid #333;
                    border-radius: 8px;
                }

                .resultsTable {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.95em;
                }

                .tableHeader {
                    background-color: #333;
                    color: #f0f0f0;
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 2px solid #61dafb;
                    font-weight: 700;
                    position: sticky; /* Keep header visible on scroll */
                    top: 0;
                    z-index: 10;
                }

                .tableCell {
                    padding: 12px 15px;
                    border-bottom: 1px solid #444;
                    color: #ccc;
                    text-align: left;
                    vertical-align: top;
                    background-color: #2a2a2a;
                }

                .resultsTable tbody tr:hover .tableCell {
                    background-color: #383838;
                }

                /* Highlight specific columns */
                .languageCell {
                    font-weight: 700;
                    text-align: center;
                }

                .scoreCell {
                    font-weight: 400;
                    color: #7aff7a; /* Green for score */
                    text-align: center;
                }

                .documentTextCell {
                    text-align: left;
                    vertical-align: middle; 
                    font-size: 0.9em;
                    /* Ensures the truncated text wraps nicely */
                    word-wrap: break-word; 
                    white-space: normal;
                }

                .errorBox {
                    background-color: #3d1c1c; /* Dark red background */
                    color: #ff6347; /* Bright red text */
                    padding: 15px;
                    border-radius: 6px;
                    border: 1px solid #ff6347;
                    margin-top: 20px;
                    font-weight: bold;
                    text-align: center;
                }

                /* Responsive adjustments */
                @media (max-width: 600px) {
                    .searchForm {
                        flex-direction: column;
                        gap: 15px;
                    }
                    .searchInput {
                        width: 100%;
                        min-width: unset;
                    }
                    .searchButton {
                        width: 100%;
                    }
                    .container {
                        padding: 15px;
                        width: 100%;
                    }
                }
                `}
            </style>
            
            {/* Component Start */}
            <div className="container"> 
                
                <h1 className="title">🌍 Multilingual Retrieval System</h1>
                <p className="introText">
                    Enter your query in **English** to retrieve documents from Hindi, Bengali, and Telugu corpora.
                </p>

                {/* --- Search Form --- */}
                <form onSubmit={handleSearch} className="searchForm">
                    
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Enter your English search query here..."
                        required
                        className="searchInput"
                    />
                    <input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(Math.max(1, parseInt(e.target.value) || 1))}
                        min="1"
                        max="50"
                        title="Number of results (Top K)"
                        className="searchInput"
                        style={{ width: '80px', flexGrow: 0 }} 
                    />
                    <button 
                        type="submit" 
                        disabled={loading || !query.trim()} 
                        className="searchButton"
                    >
                        {loading ? 'Searching...' : 'Search'}
                    </button>
                </form>

                {/* --- Loading / Error / Results Display --- */}
                
                {loading && <p style={{ color: '#61dafb', textAlign: 'center' }}>🔎 Searching for relevant documents...</p>}
                
                {error && !loading && (
                    <div className="errorBox">
                        **Error:** {error}
                    </div>
                )}

                {results.length > 0 && !loading && (
                    <>
                        <h2 style={{ color: '#f0f0f0', marginBottom: '15px' }}>Retrieved Documents ({results.length})</h2>
                        
                        <div className="resultsContainer">
                            <table className="resultsTable">
                                <thead>
                                    <tr>
                                        <th className="tableHeader" style={{width: '5%', textAlign: 'center'}}>Rank</th>
                                        <th className="tableHeader" style={{width: '10%'}}>Language</th>
                                        <th className="tableHeader" style={{width: '10%', textAlign: 'center'}}>Score</th>
                                        <th className="tableHeader" style={{width: '55%'}}>Document Text</th>
                                        <th className="tableHeader" style={{width: '20%'}}>Document ID</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {/* Map over the results array */}
                                    {results.map((result) => {
                                        // Logic to extract only the first line of text
                                        const fullText = result.text || 'Text not available';
                                        const firstLine = fullText.split('\n')[0].trim();
                                        // Check if the text actually had more than one line
                                        const isMultiLine = fullText.includes('\n');
                                        const langStyle = getLanguageStyle(result.language);

                                        return (
                                            <tr key={result.doc_id}>
                                                {/* Rank Cell */}
                                                <td className="tableCell" style={{ textAlign: 'center' }}>{result.rank}</td>
                                                
                                                {/* Language Cell: Use language badge styling */}
                                                <td className={`tableCell languageCell`}>
                                                    <span style={{ 
                                                        ...langStyle,
                                                        padding: '4px 8px',
                                                        borderRadius: '12px',
                                                        fontSize: '0.8em',
                                                        display: 'inline-block'
                                                    }}>
                                                        {result.language.toUpperCase()}
                                                    </span>
                                                </td>
                                                
                                                {/* Score Cell */}
                                                <td className={`tableCell scoreCell`} style={{ textAlign: 'center' }}>{result.score.toFixed(4)}</td>
                                                
                                                {/* Document Text Cell: Displays only the first line with ellipsis if truncated */}
                                                <td className={`tableCell documentTextCell`}> 
                                                    {firstLine}{isMultiLine && firstLine.length > 0 ? '...' : ''}
                                                </td>
                                                
                                                {/* Document ID Cell */}
                                                <td className="tableCell" style={{ fontSize: '0.7em', color: '#999' }}>{result.doc_id}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default SearchComponent;
