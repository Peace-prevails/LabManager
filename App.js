import React, { useState, useEffect, useRef } from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import { 
  FaSearch, FaSpinner, FaExclamationTriangle, FaHistory, 
  FaTrash, FaRobot, FaFlask, FaBrain, FaFileUpload, 
  FaDatabase, FaPlus, FaFileAlt, FaFolderOpen, FaServer,
  FaChartBar, FaInfoCircle, FaTimes, FaCheck, FaSync,
  FaAngleDown, FaAngleUp, FaDownload
} from 'react-icons/fa';
import axios from 'axios';

// Define server port at the beginning for easy modification
const SERVER_PORT = 5001;
const API_BASE_URL = `http://localhost:${SERVER_PORT}`;

// Global styles
const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
    padding: 0;
    background-color: #f5f7fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  
  * {
    box-sizing: border-box;
  }
`;

// Styled components
const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-image: linear-gradient(to bottom right, rgba(255, 255, 255, 0.97), rgba(255, 255, 255, 0.93));
  min-height: 100vh;
  position: relative;
  box-shadow: 0 0 40px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
`;

// Global background styles
const AppBackground = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, #8e24aa 0%, #57068c 50%, #330662 100%);
  z-index: -2;
`;

// Building elements
const BuildingElements = styled.div`
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 40vh;
  background: linear-gradient(180deg, transparent 0%, rgba(87, 6, 140, 0.2) 100%);
  z-index: -1;
  
  &::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 10%;
    width: 35%;
    height: 80%;
    background: rgba(35, 0, 60, 0.45);
    clip-path: polygon(0% 100%, 100% 100%, 90% 20%, 60% 0%, 40% 40%, 10% 30%);
  }
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    right: 5%;
    width: 45%;
    height: 60%;
    background: rgba(25, 0, 50, 0.35);
    clip-path: polygon(0% 100%, 100% 100%, 90% 30%, 70% 20%, 30% 0%, 10% 20%);
  }
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const AvatarContainer = styled.div`
  width: 120px;
  height: 120px;
  border-radius: 60px;
  background: linear-gradient(135deg, #3498db, #9b59b6);
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 1.5rem;
  border: 5px solid #fff;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  color: white;
  font-size: 3.5rem;
  overflow: hidden;
  position: relative;
  
  .brain-container {
    position: relative;
    animation: float 3s ease-in-out infinite;
  }
  
  .brain-pulse {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%);
    animation: pulse 2s ease-in-out infinite;
  }
  
  .circuit {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    opacity: 0.2;
    background: 
      linear-gradient(90deg, transparent 49%, white 49%, white 51%, transparent 51%) 0 0 / 10px 10px,
      linear-gradient(0deg, transparent 49%, white 49%, white 51%, transparent 51%) 0 0 / 10px 10px;
  }
  
  @keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
  }
  
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.2; }
    50% { transform: scale(1.1); opacity: 0.4; }
  }
`;

const BackgroundIcon = styled.div`
  position: absolute;
  bottom: -10px;
  right: -10px;
  font-size: 5rem;
  opacity: 0.1;
  color: #2c3e50;
  z-index: -1;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  
  & > svg {
    width: 60px;
    height: 60px;
    color: white;
  }
`;

const Title = styled.h1`
  color: #2c3e50;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  
  &:before, &:after {
    content: "⚛";
    margin: 0 15px;
    color: #3498db;
    opacity: 0.5;
  }
`;

const Subtitle = styled.p`
  color: #7f8c8d;
  font-size: 1.1rem;
  max-width: 80%;
  text-align: center;
`;

const TabContainer = styled.div`
  display: flex;
  margin-bottom: 2rem;
  border-bottom: 1px solid #e0e0e0;
`;

const TabButton = styled.div`
  padding: 0.8rem 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  font-weight: ${props => props.active ? '600' : '400'};
  color: ${props => props.active ? '#3498db' : '#7f8c8d'};
  border-bottom: 2px solid ${props => props.active ? '#3498db' : 'transparent'};
  
  &:hover {
    color: #3498db;
  }
  
  svg {
    margin-right: 0.5rem;
  }
`;

const QueryForm = styled.form`
  display: flex;
  flex-direction: column;
  margin-bottom: 2rem;
  background: rgba(255, 255, 255, 0.9);
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(to right, #3498db, #2980b9);
  }
`;

const QueryInput = styled.textarea`
  width: 100%;
  min-height: 100px;
  padding: 0.8rem;
  margin-bottom: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  resize: vertical;
`;

const CharCounter = styled.div`
  text-align: right;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  color: ${props => props.isOverLimit ? '#e74c3c' : '#7f8c8d'};
`;

const ButtonGroup = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
`;

const Button = styled.button`
  padding: 0.8rem 1.5rem;
  background-color: ${props => props.secondary ? '#95a5a6' : '#3498db'};
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: background-color 0.2s;

  &:hover {
    background-color: ${props => props.secondary ? '#7f8c8d' : '#2980b9'};
  }

  &:disabled {
    background-color: #bdc3c7;
    cursor: not-allowed;
  }
`;

const ButtonIcon = styled.span`
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
`;

const FileUploadContainer = styled.div`
  margin-top: 1rem;
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 4px;
  border: 1px dashed #bdc3c7;
`;

const FileUploadInput = styled.input`
  display: none;
`;

const FileUploadLabel = styled.label`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s;
  
  &:hover {
    background-color: #edf2f7;
  }
`;

const FileInfo = styled.div`
  display: flex;
  align-items: center;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background-color: #edf2f7;
  border-radius: 4px;
  justify-content: space-between;
`;

const FileIcon = styled.span`
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
  color: #3498db;
`;

const RemoveFileButton = styled.button`
  background: none;
  border: none;
  color: #e74c3c;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.25rem;
  border-radius: 50%;
  
  &:hover {
    background-color: #f8d7da;
  }
`;

const ResultSection = styled.div`
  margin-top: 2rem;
`;

const ResultCard = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, #3498db, #2980b9);
  }
`;

const ResponseHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #eee;
`;

const ResponseTitle = styled.h3`
  color: #2c3e50;
  margin: 0;
`;

const ResponseTime = styled.span`
  color: #7f8c8d;
  font-size: 0.9rem;
`;

const QueryText = styled.p`
  color: #34495e;
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 4px;
  border-left: 4px solid #3498db;
  margin-bottom: 1.5rem;
  position: relative;
  padding-left: 2.5rem;
  
  &::before {
    content: '';
    position: absolute;
    left: 10px;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 24px;
    background: #3498db;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 12px;
  }
`;

const ResponseText = styled.div`
  line-height: 1.6;
  color: #2c3e50;
  white-space: pre-wrap;
  position: relative;
  padding-left: 3rem;
  
  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    width: 36px;
    height: 36px;
    background-color: #f0f7ff;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
`;

const BotAvatar = styled.div`
  position: absolute;
  left: 0;
  top: 0;
  width: 36px;
  height: 36px;
  background-color: #edf7ff;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #3498db;
  font-size: 18px;
  box-shadow: 0 2px 10px rgba(52, 152, 219, 0.2);
  overflow: hidden;
  
  /* Animation for the bot avatar */
  .bot-face {
    display: inline-block;
    animation: pulse 2s infinite;
  }
  
  .bot-antenna {
    position: absolute;
    top: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 4px;
    height: 8px;
    background-color: #3498db;
    border-radius: 2px;
    animation: blink 1.5s infinite;
  }
  
  .bot-eyes {
    position: absolute;
    top: 10px;
    width: 100%;
    display: flex;
    justify-content: space-around;
  }
  
  .bot-eye {
    width: 8px;
    height: 8px;
    background-color: #2c3e50;
    border-radius: 50%;
    animation: blink 3s infinite;
  }
  
  .bot-mouth {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    width: 14px;
    height: 3px;
    background-color: #2c3e50;
    border-radius: 2px;
    animation: talk 1s infinite;
  }
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
  }
  
  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
  
  @keyframes talk {
    0%, 100% { width: 14px; }
    50% { width: 10px; }
  }
`;

const SourcesList = styled.div`
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid #eee;
`;

const SourcesTitle = styled.h4`
  color: #2c3e50;
  margin-top: 0;
`;

const SourceItem = styled.div`
  padding: 0.5rem 0;
  color: #7f8c8d;
  font-size: 0.9rem;
`;

const HistoryTitle = styled.h3`
  color: #2c3e50;
  margin-top: 2rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  position: relative;
  padding-bottom: 8px;
  
  &:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 3px;
    background: linear-gradient(to right, #3498db, #1abc9c);
    border-radius: 3px;
  }
`;

const HistoryIcon = styled.span`
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
`;

const HistoryList = styled.div`
  background: white;
  border-radius: 8px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
`;

const HistoryItem = styled.div`
  padding: 1rem;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #f8f9fa;
  }

  &:last-child {
    border-bottom: none;
  }
`;

const HistoryQuery = styled.p`
  margin: 0;
  color: #34495e;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  color: #7f8c8d;
  background: white;
  border-radius: 8px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  
  &:before {
    content: '📋';
    display: block;
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }
`;

const ErrorMessage = styled.div`
  background-color: #fdedee;
  color: #e74c3c;
  padding: 1rem;
  border-radius: 4px;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
`;

const ErrorIcon = styled.span`
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
`;

const LoadingState = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  color: #7f8c8d;
`;

const SpinnerIcon = styled.span`
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const Collapsible = styled.div`
  margin-top: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
`;

const CollapsibleHeader = styled.div`
  padding: 1rem;
  background-color: #f8f9fa;
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  font-weight: 600;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: #edf2f7;
  }
`;

const CollapsibleContent = styled.div`
  padding: ${props => props.isOpen ? '1rem' : '0'};
  max-height: ${props => props.isOpen ? '1000px' : '0'};
  overflow: hidden;
  transition: max-height 0.3s ease, padding 0.3s ease;
`;

const Badge = styled.span`
  background-color: ${props => props.color || '#3498db'};
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  margin-left: 0.5rem;
`;

const SystemStatusBar = styled.div`
  background-color: ${props => {
    if (props.status === 'ready') return '#d4edda';
    if (props.status === 'error') return '#f8d7da';
    return '#fff3cd'; // initializing
  }};
  color: ${props => {
    if (props.status === 'ready') return '#155724';
    if (props.status === 'error') return '#721c24';
    return '#856404'; // initializing
  }};
  padding: 0.75rem;
  border-radius: 4px;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const StatusIcon = styled.span`
  margin-right: 0.5rem;
  display: flex;
  align-items: center;
`;

const Tabs = styled.div`
  display: flex;
  margin-bottom: 2rem;
  border-bottom: 1px solid #e0e0e0;
`;

const Tab = styled.div`
  padding: 0.8rem 1.5rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  font-weight: ${props => props.active ? '600' : '400'};
  color: ${props => props.active ? '#3498db' : '#7f8c8d'};
  border-bottom: 2px solid ${props => props.active ? '#3498db' : 'transparent'};
  
  &:hover {
    color: #3498db;
  }
  
  svg {
    margin-right: 0.5rem;
  }
`;

const Card = styled.div`
  background: white;
  padding: 1.5rem;
  border-radius: 8px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
`;

const CardHeader = styled.div`
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const CardTitle = styled.h3`
  color: #2c3e50;
  margin: 0;
  display: flex;
  align-items: center;
  
  svg {
    margin-right: 0.5rem;
  }
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const StatCard = styled.div`
  background: white;
  padding: 1.25rem;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  transition: transform 0.2s, box-shadow 0.2s;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
  }
`;

const StatIcon = styled.div`
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background-color: ${props => props.color || '#3498db'};
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1rem;
  font-size: 1.5rem;
`;

const StatValue = styled.div`
  font-size: 1.75rem;
  font-weight: 700;
  color: #2c3e50;
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.div`
  color: #7f8c8d;
  font-size: 0.9rem;
`;

// Main application component
function App() {
  const [query, setQuery] = useState('');
  const [queryHistory, setQueryHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentResponse, setCurrentResponse] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [activeTab, setActiveTab] = useState('query');
  const [systemStatus, setSystemStatus] = useState({ status: 'initializing', message: 'System is starting up...' });
  const [databaseStats, setDatabaseStats] = useState(null);
  const [showDocumentSummary, setShowDocumentSummary] = useState(false);
  const [showResearchDirections, setShowResearchDirections] = useState(false);
  const [directoryPath, setDirectoryPath] = useState('');
  const [fileContent, setFileContent] = useState('');
  const [sourceLabel, setSourceLabel] = useState('');
  const [sources, setSources] = useState([]);
  const [isLoadingSources, setIsLoadingSources] = useState(false);
  const fileInputRef = useRef(null);
  
  const MAX_QUERY_LENGTH = 1000; // Maximum query length limit
  
  // Load query history from local storage
  useEffect(() => {
    const savedHistory = localStorage.getItem('queryHistory');
    if (savedHistory) {
      try {
        setQueryHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Could not parse query history:', e);
      }
    }
    
    // Check system status
    fetchSystemStatus();
    // Fetch sources on initial load
    fetchSources();
  }, []);
  
  // Save history to local storage
  useEffect(() => {
    localStorage.setItem('queryHistory', JSON.stringify(queryHistory));
  }, [queryHistory]);
  
  // Fetch system status
  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/status`);
      setSystemStatus({
        status: response.data.status,
        message: response.data.system_info.message
      });
      
      if (response.data.database_stats) {
        setDatabaseStats(response.data.database_stats);
      }
    } catch (err) {
      console.error('Error fetching system status:', err);
      setSystemStatus({
        status: 'error',
        message: 'Could not connect to the server'
      });
    }
  };

  // Fetch database stats
  const fetchDatabaseStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/documents/stats`);
      if (response.data && !response.data.error) {
        setDatabaseStats(response.data.stats);
      } else {
        console.error("Failed to fetch database stats:", response.data);
      }
    } catch (err) {
      console.error('Error fetching database stats:', err);
    }
  };

  // Fetch sources
  const fetchSources = async () => {
    setIsLoadingSources(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/documents/stats`);
      if (response.data && !response.data.error && response.data.stats && response.data.stats.sources) {
        // Format sources for display
        const sourcesList = response.data.stats.sources.map(sourceData => ({
          name: sourceData[0],
          count: sourceData[1]
        }));
        setSources(sourcesList);
      }
    } catch (err) {
      console.error('Error fetching sources:', err);
    } finally {
      setIsLoadingSources(false);
    }
  };

  // Delete a specific source
  const deleteSource = async (sourceName) => {
    if (window.confirm(`Are you sure you want to delete all documents from source: ${sourceName}?`)) {
      setIsLoading(true);
      try {
        const response = await axios.delete(`${API_BASE_URL}/api/documents`, {
          data: {
            metadata_field: 'source',
            metadata_value: sourceName
          }
        });
        
        if (response.data && !response.data.error) {
          alert(`Successfully deleted documents from source: ${sourceName}`);
          fetchSources(); // Refresh sources list
          fetchDatabaseStats(); // Update statistics
        } else {
          alert(`Error: ${response.data.message || 'Unknown error'}`);
        }
      } catch (err) {
        console.error('Error deleting source:', err);
        alert(`Error: ${err.response?.data?.message || 'Failed to delete source'}`);
      } finally {
        setIsLoading(false);
      }
    }
  };
  
  // Refresh system status and stats
  const refreshSystemStatus = async () => {
    await fetchSystemStatus();
    await fetchDatabaseStats();
    await fetchSources();
  };
  
  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };
  // Remove selected file
  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Process a standard query
  const handleStandardQuery = async () => {
    setError(null);
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/query`, { query });
      
      const newResponse = {
        id: Date.now(),
        query,
        answer: response.data.result,
        sources: response.data.sources || [],
        timestamp: new Date().toISOString(),
        processing_time: response.data.processing_time
      };
      
      setCurrentResponse(newResponse);
      setQueryHistory(prev => [newResponse, ...prev]);
      
      // Clear query after successful submission
      setQuery('');
    } catch (err) {
      console.error('Query error:', err);
      setError(err.response?.data?.message || 'An unknown error occurred. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Process query with file analysis
  const handleFileQuery = async () => {
    if (!selectedFile) {
      setError('Please select a file to analyze');
      return;
    }
    
    setError(null);
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('query', query);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/query-with-file`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      const newResponse = {
        id: Date.now(),
        query,
        answer: response.data.result,
        sources: response.data.sources || [],
        document_summary: response.data.document_summary || '',
        research_directions: response.data.research_directions || '',
        analyzed_file: response.data.analyzed_file || selectedFile.name,
        timestamp: new Date().toISOString(),
        processing_time: response.data.processing_time,
        file_analysis: true
      };
      
      setCurrentResponse(newResponse);
      setQueryHistory(prev => [newResponse, ...prev]);
      
      // Auto-open document summary for file analysis
      setShowDocumentSummary(true);
      setShowResearchDirections(true);
      
      // Clear query and file after successful submission
      setQuery('');
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      console.error('File query error:', err);
      setError(err.response?.data?.message || 'An error occurred while analyzing the file.');
    } finally {
      setIsLoading(false);
    }
  };
  
// Handle form submission
const handleSubmit = async (e) => {
  e.preventDefault();
  
  if (!query.trim() || query.length > MAX_QUERY_LENGTH) {
    return;
  }
  
  if (selectedFile) {
    await handleFileQuery();
  } else {
    await handleStandardQuery();
  }
};

// Handle history item click
const handleHistoryClick = (historyItem) => {
  setQuery(historyItem.query);
  setCurrentResponse(historyItem);
  
  // If it's a file analysis response, show the sections
  if (historyItem.file_analysis) {
    setShowDocumentSummary(true);
    setShowResearchDirections(true);
  } else {
    setShowDocumentSummary(false);
    setShowResearchDirections(false);
  }
};

// Clear history
const clearHistory = () => {
  setQueryHistory([]);
  localStorage.removeItem('queryHistory');
};

// Add document to knowledge base
const handleAddDocument = async () => {
  // Validate document content and source label
  if (!fileContent.trim()) {
    setError('Please enter document content');
    return;
  }
  
  if (!sourceLabel.trim()) {
    setError('Please enter a Document Source Label');
    return;
  }
  
  setIsLoading(true);
  setError(null);
  
  try {
    const response = await axios.post(`${API_BASE_URL}/api/documents`, {
      content: fileContent,
      source: sourceLabel.trim()
    });
    
    // Clear form fields on success
    setFileContent('');
    setSourceLabel('');
    
    // Refresh data
    await fetchDatabaseStats();
    await fetchSources();
    
    alert('Document added successfully!');
  } catch (err) {
    console.error('Error adding document:', err);
    setError(err.response?.data?.message || 'Failed to add document');
  } finally {
    setIsLoading(false);
  }
};

// Import directory
const handleImportDirectory = async () => {
  if (!directoryPath.trim()) {
    setError('Please enter a directory path');
    return;
  }
  
  setIsLoading(true);
  setError(null);
  
  // Show immediate feedback
  alert(`Starting import of directory: ${directoryPath}... Please wait.`);
  
  try {
    const response = await axios.post(`${API_BASE_URL}/api/documents/directory`, {
      directory_path: directoryPath
    });
    
    // Clear directory path on success
    setDirectoryPath('');
    
    // Refresh data
    await fetchDatabaseStats();
    await fetchSources();
    
    // Show success message
    alert(`Directory imported successfully! Added ${response.data.document_count || 0} documents (${response.data.chunk_count || 0} chunks).`);
  } catch (err) {
    console.error('Error importing directory:', err);
    setError(err.response?.data?.message || 'Failed to import directory');
    alert(`Error importing directory: ${err.response?.data?.message || 'An unknown error occurred'}`);
  } finally {
    setIsLoading(false);
  }
};

// Upload file to knowledge base
// Enhanced handleFileUpload function with better feedback
const handleFileUpload = async (e) => {
  if (!e.target.files || !e.target.files[0]) return;
  
  const file = e.target.files[0];
  const formData = new FormData();
  formData.append('file', file);
  
  setIsLoading(true);
  setError(null);
  
  // Show immediate feedback
  alert(`Starting upload of "${file.name}"... Please wait for processing to complete.`);
  
  try {
    const response = await axios.post(`${API_BASE_URL}/api/documents/file`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    // Clear input
    e.target.value = '';
    
    // Refresh data
    await fetchDatabaseStats();
    await fetchSources();
    
    if (response.data.error) {
      setError(response.data.message || 'Failed to process uploaded file');
      alert(`Error: ${response.data.message || 'Failed to process uploaded file'}`);
    } else {
      alert(`File "${file.name}" imported successfully! Added ${response.data.chunk_count || 0} chunks.`);
    }
  } catch (err) {
    console.error('Error uploading file:', err);
    setError(err.response?.data?.message || 'Failed to upload file');
    alert(`Error uploading file: ${err.response?.data?.message || 'An unknown error occurred'}`);
  } finally {
    setIsLoading(false);
  }
};

// Reset database
const handleResetDatabase = async () => {
  if (!window.confirm('Are you sure you want to reset the entire knowledge base? This action cannot be undone!')) {
    return;
  }
  
  setIsLoading(true);
  setError(null);
  
  try {
    await axios.post(`${API_BASE_URL}/api/documents/reset`);
    
    // Refresh database stats and sources
    await fetchDatabaseStats();
    await fetchSources();
    
    alert('Knowledge base has been reset successfully');
  } catch (err) {
    console.error('Error resetting database:', err);
    setError(err.response?.data?.message || 'Failed to reset database');
  } finally {
    setIsLoading(false);
  }
};

// Source management component
const SourceManagement = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <FaDatabase /> Manage Knowledge Sources
        </CardTitle>
        <Button type="button" onClick={fetchSources} disabled={isLoadingSources}>
          <ButtonIcon>{isLoadingSources ? <FaSpinner /> : <FaSync />}</ButtonIcon>
          Refresh
        </Button>
      </CardHeader>
      
      {isLoadingSources ? (
        <LoadingState>
          <SpinnerIcon><FaSpinner /></SpinnerIcon>
          Loading sources...
        </LoadingState>
      ) : sources.length > 0 ? (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #eee' }}>Source Name</th>
              <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>Chunks</th>
              <th style={{ textAlign: 'center', padding: '8px', borderBottom: '1px solid #eee' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((source) => (
              <tr key={source.name}>
                <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{source.name}</td>
                <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>{source.count}</td>
                <td style={{ textAlign: 'center', padding: '8px', borderBottom: '1px solid #eee' }}>
                  <Button
                    type="button"
                    style={{ backgroundColor: '#e74c3c', padding: '4px 8px' }}
                    onClick={() => deleteSource(source.name)}
                  >
                    <FaTrash />
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <EmptyState>No sources found in knowledge base</EmptyState>
      )}
    </Card>
  );
};

// Render the query tab content
const renderQueryTab = () => (
  <>
    <QueryForm onSubmit={handleSubmit}>
      <QueryInput 
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter your question, e.g.: How does MRI work? What MRI equipment does this lab have?"
        disabled={isLoading}
      />
      
      <CharCounter isOverLimit={query.length > MAX_QUERY_LENGTH}>
        {query.length}/{MAX_QUERY_LENGTH}
        {query.length > MAX_QUERY_LENGTH && ' Character limit exceeded'}
      </CharCounter>
      
      {/* File upload section */}
      <FileUploadContainer>
        <FileUploadLabel htmlFor="file-upload">
          <FileIcon><FaFileUpload /></FileIcon>
          {selectedFile ? 'Change file' : 'Upload a file for analysis (optional)'}
        </FileUploadLabel>
        <FileUploadInput 
          id="file-upload" 
          type="file" 
          onChange={handleFileChange}
          ref={fileInputRef}
          disabled={isLoading}
        />
        
        {selectedFile && (
          <FileInfo>
            <div>
              <FileIcon><FaFileAlt /></FileIcon>
              {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
            </div>
            <RemoveFileButton onClick={handleRemoveFile} title="Remove file">
              <FaTimes />
            </RemoveFileButton>
          </FileInfo>
        )}
        
        <div style={{ fontSize: '0.8rem', color: '#7f8c8d', marginTop: '0.5rem' }}>
          Note: Uploaded files are analyzed but not added to the knowledge base
        </div>
      </FileUploadContainer>
      
      <ButtonGroup>
        <Button 
          type="submit" 
          disabled={!query.trim() || query.length > MAX_QUERY_LENGTH || isLoading}
        >
          <ButtonIcon>
            {isLoading ? <FaSpinner /> : <FaSearch />}
          </ButtonIcon>
          {isLoading ? 'Processing...' : (selectedFile ? 'Analyze File & Query' : 'Submit Query')}
        </Button>
        
        <Button 
          type="button" 
          secondary 
          onClick={() => {
            setQuery('');
            handleRemoveFile();
          }}
          disabled={(!query && !selectedFile) || isLoading}
        >
          <ButtonIcon><FaTrash /></ButtonIcon>
          Clear All
        </Button>
      </ButtonGroup>
      
      {error && (
        <ErrorMessage>
          <ErrorIcon><FaExclamationTriangle /></ErrorIcon>
          {error}
        </ErrorMessage>
      )}
    </QueryForm>
    
    {isLoading && (
      <LoadingState>
        <SpinnerIcon><FaSpinner /></SpinnerIcon>
        Processing your query, please wait...
      </LoadingState>
    )}
    
    {currentResponse && !isLoading && (
      <ResultSection>
        <ResultCard>
          <ResponseHeader>
            <ResponseTitle>
              {currentResponse.file_analysis 
                ? `File Analysis: ${currentResponse.analyzed_file}`
                : 'Query Result'}
            </ResponseTitle>
            <ResponseTime>
              {new Date(currentResponse.timestamp).toLocaleString()} 
              {currentResponse.processing_time && `(${currentResponse.processing_time})`}
            </ResponseTime>
          </ResponseHeader>
          
          <QueryText>{currentResponse.query}</QueryText>
          
          <ResponseText>
            <BotAvatar>
              <div className="bot-antenna"></div>
              <div className="bot-eyes">
                <div className="bot-eye"></div>
                <div className="bot-eye"></div>
              </div>
              <div className="bot-mouth"></div>
            </BotAvatar>
            {currentResponse.answer ? currentResponse.answer.trim() : ""}
          </ResponseText>
          
          {/* Document Summary Collapsible (for file analysis) */}
          {currentResponse.document_summary && (
            <Collapsible>
              <CollapsibleHeader onClick={() => setShowDocumentSummary(!showDocumentSummary)}>
                Document Summary
                {showDocumentSummary ? <FaAngleUp /> : <FaAngleDown />}
              </CollapsibleHeader>
              <CollapsibleContent isOpen={showDocumentSummary}>
                {currentResponse.document_summary}
              </CollapsibleContent>
            </Collapsible>
          )}
          
          {/* Research Directions Collapsible (for file analysis) */}
          {currentResponse.research_directions && (
            <Collapsible>
              <CollapsibleHeader onClick={() => setShowResearchDirections(!showResearchDirections)}>
                Research Directions
                {showResearchDirections ? <FaAngleUp /> : <FaAngleDown />}
              </CollapsibleHeader>
              <CollapsibleContent isOpen={showResearchDirections}>
                {currentResponse.research_directions}
              </CollapsibleContent>
            </Collapsible>
          )}
          
          {currentResponse.sources && currentResponse.sources.length > 0 && (
            <SourcesList>
              <SourcesTitle>Reference Sources:</SourcesTitle>
              {currentResponse.sources.map((source, index) => (
                <SourceItem key={index}>
                  {index + 1}. {source}
                </SourceItem>
              ))}
            </SourcesList>
          )}
        </ResultCard>
      </ResultSection>
    )}
    
    <HistoryTitle>
      <HistoryIcon><FaHistory /></HistoryIcon>
      Query History
      {queryHistory.length > 0 && (
        <Button 
          type="button" 
          secondary 
          style={{ marginLeft: 'auto', padding: '0.4rem 0.8rem', fontSize: '0.9rem' }}
          onClick={clearHistory}
        >
          <ButtonIcon><FaTrash /></ButtonIcon>
          Clear History
        </Button>
      )}
    </HistoryTitle>
    
    {queryHistory.length > 0 ? (
      <HistoryList>
        {queryHistory.map(item => (
          <HistoryItem key={item.id} onClick={() => handleHistoryClick(item)}>
            <HistoryQuery>
              {item.file_analysis && <Badge color="#9b59b6">File Analysis</Badge>}
              {item.query}
            </HistoryQuery>
          </HistoryItem>
        ))}
      </HistoryList>
    ) : (
      <EmptyState>
        No query history yet
      </EmptyState>
    )}
  </>
);

// Render knowledge base tab content
const renderKnowledgeBaseTab = () => (
  <>
    <Card>
      <CardHeader>
        <CardTitle>
          <FaDatabase /> Knowledge Base Statistics
        </CardTitle>
        <Button type="button" onClick={refreshSystemStatus}>
          <ButtonIcon><FaSync /></ButtonIcon>
          Refresh
        </Button>
      </CardHeader>
      
      {databaseStats ? (
        <Grid>
          <StatCard>
            <StatIcon color="#3498db"><FaFileAlt /></StatIcon>
            <StatValue>{databaseStats.document_count || 0}</StatValue>
            <StatLabel>Text Chunks</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatIcon color="#2ecc71"><FaServer /></StatIcon>
            <StatValue>{databaseStats.sources?.length || 0}</StatValue>
            <StatLabel>Sources</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatIcon color="#9b59b6"><FaFlask /></StatIcon>
            <StatValue>{databaseStats.status === "active" ? "Active" : "Inactive"}</StatValue>
            <StatLabel>Database Status</StatLabel>
          </StatCard>
        </Grid>
      ) : (
        <LoadingState>
          <SpinnerIcon><FaSpinner /></SpinnerIcon>
          Loading database statistics...
        </LoadingState>
      )}
    </Card>
    
    {/* Source Management */}
    <SourceManagement />
    
    <Card>
      <CardHeader>
        <CardTitle>
          <FaPlus /> Add Document
        </CardTitle>
      </CardHeader>
      
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Document Source Label:</label>
        <input 
          type="text" 
          value={sourceLabel}
          onChange={(e) => setSourceLabel(e.target.value)}
          placeholder="E.g., Lab Report 2023"
          style={{
            width: '100%',
            padding: '0.8rem',
            borderRadius: '4px',
            border: '1px solid #ddd'
          }}
        />
      </div>
      
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Document Content:</label>
        <textarea
          value={fileContent}
          onChange={(e) => setFileContent(e.target.value)}
          placeholder="Enter document text here..."
          style={{
            width: '100%',
            padding: '0.8rem',
            borderRadius: '4px',
            border: '1px solid #ddd',
            minHeight: '150px'
          }}
        />
      </div>
      
      <Button 
        type="button" 
        onClick={handleAddDocument}
        disabled={!fileContent.trim() || !sourceLabel.trim() || isLoading}
      >
        <ButtonIcon>
          {isLoading ? <FaSpinner /> : <FaPlus />}
        </ButtonIcon>
        Add to Knowledge Base
      </Button>
      
      {error && (
        <ErrorMessage>
          <ErrorIcon><FaExclamationTriangle /></ErrorIcon>
          {error}
        </ErrorMessage>
      )}
    </Card>
    
    <Card>
      <CardHeader>
        <CardTitle>
          <FaFileUpload /> Upload File to Knowledge Base
        </CardTitle>
      </CardHeader>
      
      <div style={{ marginBottom: '1rem' }}>
        <FileUploadContainer>
          <FileUploadLabel htmlFor="kb-file-upload">
            <FileIcon><FaFileUpload /></FileIcon>
            Click to select a file to add to the knowledge base
          </FileUploadLabel>
          <FileUploadInput 
            id="kb-file-upload" 
            type="file" 
            onChange={handleFileUpload}
            disabled={isLoading}
          />
        </FileUploadContainer>
      </div>
    </Card>
    
    <Card>
      <CardHeader>
        <CardTitle>
          <FaFolderOpen /> Import Directory
        </CardTitle>
      </CardHeader>
      
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>Directory Path:</label>
        <input 
          type="text" 
          value={directoryPath}
          onChange={(e) => setDirectoryPath(e.target.value)}
          placeholder="/path/to/documents"
          style={{
            width: '100%',
            padding: '0.8rem',
            borderRadius: '4px',
            border: '1px solid #ddd'
          }}
        />
      </div>
      
      <Button 
        type="button" 
        onClick={handleImportDirectory}
        disabled={!directoryPath.trim() || isLoading}
      >
        <ButtonIcon>
          {isLoading ? <FaSpinner /> : <FaFolderOpen />}
        </ButtonIcon>
        Import Directory
      </Button>
    </Card>
    
    <Card>
      <CardHeader>
        <CardTitle style={{ color: '#e74c3c' }}>
          <FaTrash /> Reset Knowledge Base
        </CardTitle>
      </CardHeader>
      
      <div style={{ marginBottom: '1rem' }}>
        <p>This will permanently delete all documents in the knowledge base. This action cannot be undone.</p>
      </div>
      
      <Button 
        type="button" 
        onClick={handleResetDatabase}
        disabled={isLoading}
        style={{ backgroundColor: '#e74c3c' }}
      >
        <ButtonIcon><FaTrash /></ButtonIcon>
        Reset Knowledge Base
      </Button>
    </Card>
  </>
);

return (
  <>
    <GlobalStyle />
    <AppBackground />
    <BuildingElements />
    <Container>
      <BackgroundIcon>
        <FaFlask />
      </BackgroundIcon>
      <Header>
        <AvatarContainer>
          <div className="circuit"></div>
          <div className="brain-pulse"></div>
          <div className="brain-container">
            <Logo>
              <FaBrain />
            </Logo>
          </div>
        </AvatarContainer>
        <Title>Iangone</Title>
        <Subtitle>Intelligent document retrieval and Q&A powered by Llama-2 7B with RAG</Subtitle>
      </Header>
      
      {/* System Status Bar */}
      <SystemStatusBar status={systemStatus.status}>
        <div>
          <StatusIcon>
            {systemStatus.status === 'ready' ? <FaCheck /> : 
             systemStatus.status === 'error' ? <FaExclamationTriangle /> : 
             <FaSpinner />}
          </StatusIcon>
          System Status: {systemStatus.status.toUpperCase()} - {systemStatus.message}
        </div>
        <Button type="button" onClick={refreshSystemStatus} secondary style={{ padding: '0.4rem 0.8rem' }}>
          <FaSync />
        </Button>
      </SystemStatusBar>
      
      {/* Main Tabs */}
      <TabContainer>
        <Tab active={activeTab === 'query'} onClick={() => setActiveTab('query')}>
          <FaSearch /> Query
        </Tab>
        <Tab active={activeTab === 'knowledgeBase'} onClick={() => setActiveTab('knowledgeBase')}>
          <FaDatabase /> Knowledge Base
        </Tab>
      </TabContainer>
      
      {/* Tab Content */}
      {activeTab === 'query' ? renderQueryTab() : renderKnowledgeBaseTab()}
    </Container>
  </>
);
}

export default App;
