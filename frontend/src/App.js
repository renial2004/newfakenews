import { useState, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { Shield, AlertTriangle, XCircle, TrendingUp, FileText, Link as LinkIcon, Upload, History, BarChart3, Sparkles, Moon, Sun, Newspaper, Zap, Search, Award, Target, Activity } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const themes = {
  neon: {
    name: "Neon Tech",
    colors: {
      primary: "#00f0ff",
      secondary: "#ff00ff",
      background: "#0a0a0f",
      card: "rgba(20, 20, 30, 0.8)",
      accent: "#7000ff"
    }
  },
  newspaper: {
    name: "Newspaper",
    colors: {
      primary: "#2c3e50",
      secondary: "#e74c3c",
      background: "#f5f5f0",
      card: "rgba(255, 255, 255, 0.95)",
      accent: "#34495e"
    }
  },
  cyberpunk: {
    name: "Cyberpunk",
    colors: {
      primary: "#f72585",
      secondary: "#ffd60a",
      background: "#000814",
      card: "rgba(13, 0, 26, 0.85)",
      accent: "#4cc9f0"
    }
  },
  dark: {
    name: "Dark Mode",
    colors: {
      primary: "#6366f1",
      secondary: "#8b5cf6",
      background: "#0f0f1a",
      card: "rgba(30, 30, 45, 0.9)",
      accent: "#a78bfa"
    }
  }
};

function App() {
  const [currentTheme, setCurrentTheme] = useState("cyberpunk");
  const [activeTab, setActiveTab] = useState("analyze");
  const [inputText, setInputText] = useState("");
  const [inputUrl, setInputUrl] = useState("");
  const [inputMode, setInputMode] = useState("text"); // text, url, file
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [showAssistant, setShowAssistant] = useState(true);

  const theme = themes[currentTheme];

  useEffect(() => {
    document.documentElement.style.setProperty('--theme-primary', theme.colors.primary);
    document.documentElement.style.setProperty('--theme-secondary', theme.colors.secondary);
    document.documentElement.style.setProperty('--theme-background', theme.colors.background);
    document.documentElement.style.setProperty('--theme-card', theme.colors.card);
    document.documentElement.style.setProperty('--theme-accent', theme.colors.accent);
  }, [currentTheme, theme]);

  useEffect(() => {
    loadHistory();
    loadStats();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await axios.get(`${API}/history`);
      setHistory(response.data);
    } catch (e) {
      console.error("Error loading history:", e);
    }
  };

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
    } catch (e) {
      console.error("Error loading stats:", e);
    }
  };

  const handleAnalyze = async () => {
    if (!inputText && !inputUrl) {
      toast.error("Please provide text or URL to analyze");
      return;
    }

    setAnalyzing(true);
    setResult(null);

    try {
      const response = await axios.post(`${API}/analyze`, {
        text: inputMode === "text" ? inputText : null,
        url: inputMode === "url" ? inputUrl : null
      });

      setResult(response.data);
      toast.success("Analysis completed!");
      loadHistory();
      loadStats();
    } catch (e) {
      console.error("Analysis error:", e);
      toast.error("Analysis failed. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type === "text/plain") {
      const reader = new FileReader();
      reader.onload = (event) => {
        setInputText(event.target.result);
        setInputMode("text");
        toast.success("File loaded successfully");
      };
      reader.readAsText(file);
    } else {
      toast.error("Please upload a text file");
    }
  };

  const saveStatus = async (status) => {
    if (!result) return;
    try {
      await axios.post(`${API}/save-status/${result.id}`, { status });
      toast.success(`Marked as ${status}`);
      loadHistory();
    } catch (e) {
      console.error("Save error:", e);
    }
  };

  const getVerificationIcon = (classification) => {
    switch (classification) {
      case "Verified":
        return <Shield className="w-6 h-6" data-testid="verified-icon" />;
      case "Suspicious":
        return <AlertTriangle className="w-6 h-6" data-testid="suspicious-icon" />;
      case "Fake":
        return <XCircle className="w-6 h-6" data-testid="fake-icon" />;
      default:
        return null;
    }
  };

  const getVerificationColor = (classification) => {
    switch (classification) {
      case "Verified":
        return "from-green-400 to-emerald-500";
      case "Suspicious":
        return "from-yellow-400 to-orange-500";
      case "Fake":
        return "from-red-400 to-rose-600";
      default:
        return "from-gray-400 to-gray-500";
    }
  };

  return (
    <div className="app-container" style={{ background: theme.colors.background }}>
      <Toaster position="top-center" richColors />
      
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon" style={{ background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary})` }}>
              <Shield className="w-8 h-8" />
            </div>
            <div>
              <h1 className="app-title" data-testid="app-title">MisinfoWatch</h1>
              <p className="app-subtitle">AI-Powered Fake News Detection</p>
            </div>
          </div>

          {/* Theme Switcher */}
          <div className="theme-switcher">
            {Object.keys(themes).map((key) => (
              <button
                key={key}
                data-testid={`theme-${key}-btn`}
                onClick={() => setCurrentTheme(key)}
                className={`theme-btn ${currentTheme === key ? 'active' : ''}`}
                style={{
                  background: currentTheme === key ? `linear-gradient(135deg, ${themes[key].colors.primary}, ${themes[key].colors.secondary})` : 'transparent',
                  border: `2px solid ${themes[key].colors.primary}`
                }}
                title={themes[key].name}
              >
                {key === 'neon' && <Zap className="w-4 h-4" />}
                {key === 'newspaper' && <Newspaper className="w-4 h-4" />}
                {key === 'cyberpunk' && <Sparkles className="w-4 h-4" />}
                {key === 'dark' && <Moon className="w-4 h-4" />}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="tabs-list" style={{ background: theme.colors.card }}>
            <TabsTrigger value="analyze" data-testid="tab-analyze" className="tab-trigger">
              <Search className="w-4 h-4 mr-2" />
              Analyze
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history" className="tab-trigger">
              <History className="w-4 h-4 mr-2" />
              History
            </TabsTrigger>
            <TabsTrigger value="dashboard" data-testid="tab-dashboard" className="tab-trigger">
              <BarChart3 className="w-4 h-4 mr-2" />
              Dashboard
            </TabsTrigger>
          </TabsList>

          {/* Analyze Tab */}
          <TabsContent value="analyze" className="tab-content">
            <div className="analyze-grid">
              {/* Input Section */}
              <Card className="input-card glass-effect" style={{ background: theme.colors.card }}>
                <div className="card-header">
                  <h2 className="card-title" data-testid="input-section-title">
                    <FileText className="w-5 h-5" style={{ color: theme.colors.primary }} />
                    Input Content
                  </h2>
                  
                  {/* Input Mode Selector */}
                  <div className="input-mode-selector">
                    <Button
                      data-testid="mode-text-btn"
                      variant={inputMode === "text" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setInputMode("text")}
                      style={inputMode === "text" ? { background: theme.colors.primary } : {}}
                    >
                      <FileText className="w-4 h-4 mr-1" />
                      Text
                    </Button>
                    <Button
                      data-testid="mode-url-btn"
                      variant={inputMode === "url" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setInputMode("url")}
                      style={inputMode === "url" ? { background: theme.colors.primary } : {}}
                    >
                      <LinkIcon className="w-4 h-4 mr-1" />
                      URL
                    </Button>
                    <Button
                      data-testid="mode-file-btn"
                      variant={inputMode === "file" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setInputMode("file")}
                      style={inputMode === "file" ? { background: theme.colors.primary } : {}}
                    >
                      <Upload className="w-4 h-4 mr-1" />
                      File
                    </Button>
                  </div>
                </div>

                <div className="card-body">
                  {inputMode === "text" && (
                    <Textarea
                      data-testid="input-text-area"
                      placeholder="Paste article text or news content here..."
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      className="input-textarea"
                      rows={12}
                      style={{ background: 'rgba(0,0,0,0.2)', color: currentTheme === 'newspaper' ? '#000' : '#fff' }}
                    />
                  )}

                  {inputMode === "url" && (
                    <Input
                      data-testid="input-url"
                      type="url"
                      placeholder="https://example.com/article"
                      value={inputUrl}
                      onChange={(e) => setInputUrl(e.target.value)}
                      className="input-url"
                      style={{ background: 'rgba(0,0,0,0.2)', color: currentTheme === 'newspaper' ? '#000' : '#fff' }}
                    />
                  )}

                  {inputMode === "file" && (
                    <div className="file-upload-zone" data-testid="file-upload-zone">
                      <Upload className="w-12 h-12 mb-4" style={{ color: theme.colors.primary }} />
                      <p className="mb-4">Drag and drop article file</p>
                      <Input
                        data-testid="file-input"
                        type="file"
                        accept=".txt"
                        onChange={handleFileUpload}
                        className="file-input"
                      />
                    </div>
                  )}

                  <Button
                    data-testid="analyze-btn"
                    onClick={handleAnalyze}
                    disabled={analyzing || (!inputText && !inputUrl)}
                    className="analyze-button"
                    style={{ background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary})` }}
                  >
                    {analyzing ? (
                      <>
                        <Activity className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search className="w-5 h-5 mr-2" />
                        Analyze Content
                      </>
                    )}
                  </Button>
                </div>
              </Card>

              {/* Assistant Avatar */}
              {showAssistant && (
                <div className="assistant-avatar" data-testid="assistant-avatar">
                  <div className="avatar-glow" style={{ background: `radial-gradient(circle, ${theme.colors.primary}, transparent)` }}></div>
                  <div className="avatar-icon" style={{ background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary})` }}>
                    <Sparkles className="w-8 h-8" />
                  </div>
                  <div className="avatar-message" style={{ background: theme.colors.card }}>
                    <p>Hi! I'm your AI assistant. I'll help you detect fake news using advanced NLP and machine learning.</p>
                  </div>
                </div>
              )}

              {/* Results Section */}
              {result && (
                <Card className="result-card glass-effect" style={{ background: theme.colors.card }} data-testid="result-card">
                  {/* Crisis Alert */}
                  {result.crisis_flag && (
                    <Alert className="crisis-alert" variant="destructive" data-testid="crisis-alert">
                      <AlertTriangle className="w-4 h-4" />
                      <AlertDescription>{result.crisis_reason}</AlertDescription>
                    </Alert>
                  )}

                  {/* Verification Badge */}
                  <div className="verification-badge">
                    <div className={`badge-glow bg-gradient-to-r ${getVerificationColor(result.classification)}`}>
                      {getVerificationIcon(result.classification)}
                    </div>
                    <div>
                      <h3 className="verification-title" data-testid="verification-result">{result.classification}</h3>
                      <p className="verification-subtitle">AI Analysis Complete</p>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="confidence-section">
                    <div className="confidence-header">
                      <span>Confidence Score</span>
                      <span className="confidence-value" data-testid="confidence-score">{(result.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <Progress 
                      value={result.confidence * 100} 
                      className="confidence-bar"
                      style={{ '--progress-color': theme.colors.primary }}
                    />
                  </div>

                  {/* Category & Sentiment */}
                  <div className="metadata-badges">
                    <Badge className="category-badge" data-testid="category-badge" style={{ background: theme.colors.primary }}>
                      <Target className="w-3 h-3 mr-1" />
                      {result.category}
                    </Badge>
                    <Badge className="sentiment-badge" data-testid="sentiment-badge" style={{ background: theme.colors.secondary }}>
                      <TrendingUp className="w-3 h-3 mr-1" />
                      {result.sentiment}
                    </Badge>
                    <Badge className="sensationalism-badge" data-testid="sensationalism-badge" style={{ background: theme.colors.accent }}>
                      Sensationalism: {(result.sensationalism_score * 100).toFixed(0)}%
                    </Badge>
                  </div>

                  {/* Info Panels */}
                  <div className="info-panels">
                    <div className="info-panel" style={{ borderLeft: `3px solid ${theme.colors.primary}` }}>
                      <h4 className="info-title">Why Flagged?</h4>
                      <p className="info-text" data-testid="explanation-text">{result.explanation}</p>
                    </div>

                    <div className="info-panel" style={{ borderLeft: `3px solid ${theme.colors.secondary}` }}>
                      <h4 className="info-title">Linguistic Patterns</h4>
                      <ul className="pattern-list">
                        {result.linguistic_patterns.map((pattern, idx) => (
                          <li key={idx} data-testid={`pattern-${idx}`}>• {pattern}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="info-panel" style={{ borderLeft: `3px solid ${theme.colors.accent}` }}>
                      <h4 className="info-title">Summary</h4>
                      <p className="info-text" data-testid="summary-text">{result.summary}</p>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="action-buttons">
                    <Button 
                      data-testid="save-verified-btn"
                      onClick={() => saveStatus("Verified")} 
                      variant="outline"
                      style={{ borderColor: '#10b981', color: '#10b981' }}
                    >
                      <Shield className="w-4 h-4 mr-2" />
                      Mark Verified
                    </Button>
                    <Button 
                      data-testid="save-review-btn"
                      onClick={() => saveStatus("Needs Review")} 
                      variant="outline"
                      style={{ borderColor: '#f59e0b', color: '#f59e0b' }}
                    >
                      <AlertTriangle className="w-4 h-4 mr-2" />
                      Needs Review
                    </Button>
                    <Button 
                      data-testid="save-fake-btn"
                      onClick={() => saveStatus("Fake")} 
                      variant="outline"
                      style={{ borderColor: '#ef4444', color: '#ef4444' }}
                    >
                      <XCircle className="w-4 h-4 mr-2" />
                      Mark Fake
                    </Button>
                  </div>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="tab-content">
            <Card className="history-card glass-effect" style={{ background: theme.colors.card }}>
              <div className="card-header">
                <h2 className="card-title">
                  <History className="w-5 h-5" style={{ color: theme.colors.primary }} />
                  Analysis History
                </h2>
              </div>
              <ScrollArea className="history-scroll">
                {history.length === 0 ? (
                  <div className="empty-state" data-testid="empty-history">
                    <History className="w-16 h-16 mb-4" style={{ color: theme.colors.primary, opacity: 0.3 }} />
                    <p>No analysis history yet</p>
                  </div>
                ) : (
                  <div className="history-list">
                    {history.map((item, idx) => (
                      <div key={item.id} className="history-item" data-testid={`history-item-${idx}`} style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <div className="history-badge" style={{ background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.secondary})` }}>
                          {getVerificationIcon(item.classification)}
                        </div>
                        <div className="history-content">
                          <div className="history-header">
                            <Badge style={{ background: theme.colors.primary }}>{item.classification}</Badge>
                            <span className="history-time">{new Date(item.timestamp).toLocaleString()}</span>
                          </div>
                          <p className="history-summary">{item.summary}</p>
                          <div className="history-meta">
                            <span>{item.category}</span>
                            <span>•</span>
                            <span>{(item.confidence * 100).toFixed(0)}% confidence</span>
                            {item.saved_status && (
                              <>
                                <span>•</span>
                                <Badge variant="outline">{item.saved_status}</Badge>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </Card>
          </TabsContent>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="tab-content">
            <div className="dashboard-grid">
              {/* Stats Cards */}
              <Card className="stat-card glass-effect" style={{ background: theme.colors.card }} data-testid="stat-total">
                <div className="stat-icon" style={{ background: theme.colors.primary }}>
                  <BarChart3 className="w-6 h-6" />
                </div>
                <div className="stat-content">
                  <p className="stat-label">Total Analyses</p>
                  <p className="stat-value">{stats?.total_analyses || 0}</p>
                </div>
              </Card>

              <Card className="stat-card glass-effect" style={{ background: theme.colors.card }} data-testid="stat-verified">
                <div className="stat-icon" style={{ background: '#10b981' }}>
                  <Shield className="w-6 h-6" />
                </div>
                <div className="stat-content">
                  <p className="stat-label">Verified</p>
                  <p className="stat-value">{stats?.verified_count || 0}</p>
                </div>
              </Card>

              <Card className="stat-card glass-effect" style={{ background: theme.colors.card }} data-testid="stat-suspicious">
                <div className="stat-icon" style={{ background: '#f59e0b' }}>
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div className="stat-content">
                  <p className="stat-label">Suspicious</p>
                  <p className="stat-value">{stats?.suspicious_count || 0}</p>
                </div>
              </Card>

              <Card className="stat-card glass-effect" style={{ background: theme.colors.card }} data-testid="stat-fake">
                <div className="stat-icon" style={{ background: '#ef4444' }}>
                  <XCircle className="w-6 h-6" />
                </div>
                <div className="stat-content">
                  <p className="stat-label">Fake News</p>
                  <p className="stat-value">{stats?.fake_count || 0}</p>
                </div>
              </Card>

              {/* Category Distribution */}
              <Card className="category-card glass-effect" style={{ background: theme.colors.card }}>
                <div className="card-header">
                  <h2 className="card-title">
                    <Target className="w-5 h-5" style={{ color: theme.colors.primary }} />
                    Category Distribution
                  </h2>
                </div>
                <div className="category-list">
                  {stats?.categories && Object.entries(stats.categories).map(([category, count]) => (
                    <div key={category} className="category-item" data-testid={`category-${category}`}>
                      <span className="category-name">{category}</span>
                      <div className="category-bar-container">
                        <div 
                          className="category-bar" 
                          style={{ 
                            width: `${(count / stats.total_analyses * 100).toFixed(0)}%`,
                            background: `linear-gradient(90deg, ${theme.colors.primary}, ${theme.colors.secondary})`
                          }}
                        ></div>
                      </div>
                      <span className="category-count">{count}</span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Achievements */}
              <Card className="achievement-card glass-effect" style={{ background: theme.colors.card }}>
                <div className="card-header">
                  <h2 className="card-title">
                    <Award className="w-5 h-5" style={{ color: theme.colors.primary }} />
                    Achievements
                  </h2>
                </div>
                <div className="achievement-grid">
                  <div className="achievement-badge" data-testid="achievement-beginner">
                    <div className="achievement-icon" style={{ background: '#3b82f6' }}>
                      <Target className="w-6 h-6" />
                    </div>
                    <p className="achievement-name">Beginner Analyst</p>
                    <p className="achievement-desc">Analyze your first article</p>
                  </div>
                  <div className="achievement-badge" data-testid="achievement-detective">
                    <div className="achievement-icon" style={{ background: '#8b5cf6' }}>
                      <Search className="w-6 h-6" />
                    </div>
                    <p className="achievement-name">Fact Detective</p>
                    <p className="achievement-desc">Analyze 10 articles</p>
                  </div>
                  <div className="achievement-badge" data-testid="achievement-guardian">
                    <div className="achievement-icon" style={{ background: '#10b981' }}>
                      <Shield className="w-6 h-6" />
                    </div>
                    <p className="achievement-name">Truth Guardian</p>
                    <p className="achievement-desc">Spot 5 fake news</p>
                  </div>
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;