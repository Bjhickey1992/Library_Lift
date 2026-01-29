"""
MCP (Model Context Protocol) Server for Film Library Recommendations
Exposes tools and context to LLMs for chatbot functionality
"""

import json
from typing import Dict, List, Optional, Any
from chatbot_agent import ChatbotAgent


class FilmLibraryMCPServer:
    """
    MCP Server that exposes film library recommendation tools.
    Can be used with MCP-compatible clients or directly with OpenAI function calling.
    """
    
    def __init__(self, studio_name: str = "Lionsgate"):
        self.chatbot_agent = ChatbotAgent(studio_name=studio_name)
        self.studio_name = studio_name
    
    def get_tools_schema(self) -> List[Dict]:
        """Return MCP tools schema for film recommendations."""
        return [
            {
                "name": "get_film_recommendations",
                "description": "Get film library recommendations for a specific territory based on upcoming exhibition schedules. Returns 3-5 best matches in the 0.5-0.7 similarity range (unintuitive but logical connections).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "territory": {
                            "type": "string",
                            "description": "Country code: US, UK, FR, CA, or MX",
                            "enum": ["US", "UK", "FR", "CA", "MX"]
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity score (default: 0.5)",
                            "default": 0.5,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "max_similarity": {
                            "type": "number",
                            "description": "Maximum similarity score (default: 0.7)",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of recommendations (default: 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["territory"]
                }
            },
            {
                "name": "parse_recommendation_query",
                "description": "Parse a natural language query about film recommendations and extract territory and parameters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "User's natural language query about film recommendations"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool call."""
        if tool_name == "get_film_recommendations":
            territory = arguments.get("territory", "US")
            min_sim = arguments.get("min_similarity", 0.5)
            max_sim = arguments.get("max_similarity", 0.7)
            top_n = arguments.get("top_n", 5)
            
            recommendations = self.chatbot_agent.get_recommendations(
                territory=territory,
                min_similarity=min_sim,
                max_similarity=max_sim,
                top_n=top_n
            )
            
            return {
                "territory": territory,
                "recommendations": recommendations,
                "count": len(recommendations)
            }
        
        elif tool_name == "parse_recommendation_query":
            query = arguments.get("query", "")
            result = self.chatbot_agent.get_recommendations_for_query(query)
            return result
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def get_context(self, query: str) -> Dict[str, Any]:
        """Get context for a query (library info, available territories, etc.)."""
        try:
            library_df = self.chatbot_agent._load_library()
            exhibitions_df = self.chatbot_agent._load_exhibitions()
            
            available_territories = exhibitions_df["country"].unique().tolist()
            
            return {
                "studio": self.studio_name,
                "library_size": len(library_df),
                "exhibition_films": len(exhibitions_df),
                "available_territories": available_territories,
                "query": query
            }
        except Exception as e:
            return {"error": str(e)}


# OpenAI Function Calling Format (for direct integration)
def get_openai_functions() -> List[Dict]:
    """Return OpenAI function calling format for recommendations."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_film_recommendations",
                "description": "Get film library recommendations for a territory. Returns 3-5 best matches focusing on unintuitive but logical connections (similarity 0.5-0.7).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "territory": {
                            "type": "string",
                            "description": "Country code: US, UK, FR, CA, or MX",
                            "enum": ["US", "UK", "FR", "CA", "MX"]
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of recommendations (3-5)",
                            "minimum": 3,
                            "maximum": 5
                        }
                    },
                    "required": ["territory"]
                }
            }
        }
    ]


def call_openai_function(name: str, arguments: Dict) -> Dict:
    """Call OpenAI function and return result."""
    server = FilmLibraryMCPServer()
    return server.call_tool(name, arguments)
