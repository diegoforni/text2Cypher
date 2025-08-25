"""Standalone multi-agent prototype used during early experimentation."""

import re
import json
import requests
from neo4j import GraphDatabase
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time

from config import NEO4J_DB

# CONFIG
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"
OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o"

WRITE_KEYWORDS = re.compile(r"\b(CREATE|DELETE|SET|MERGE|DROP|REMOVE|FOREACH|CALL\s+apoc\.create)\b", re.I)

@dataclass
class AgentMessage:
    """Represents a message in the multi-agent conversation"""
    agent: str
    content: str
    timestamp: float
    metadata: Optional[Dict] = None

@dataclass
class QueryAttempt:
    """Represents a query generation attempt with results"""
    query: str
    success: bool
    error: Optional[str] = None
    results: Optional[List] = None
    agent: str = ""

class CypherAgent:
    """Base class for all Cypher generation agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.conversation_history: List[AgentMessage] = []
    
    def add_message(self, content: str, metadata: Optional[Dict] = None):
        """Add a message to the agent's conversation history"""
        self.conversation_history.append(
            AgentMessage(self.name, content, time.time(), metadata)
        )
    
    def call_llm(self, prompt: str, system_message: str) -> str:
        """Make API call to OpenAI"""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0
        }
        
        try:
            print(f"🤖 [{self.name}] Calling LLM...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            
            print(f"💭 [{self.name}] LLM Response: {result[:100]}..." if len(result) > 100 else f"💭 [{self.name}] LLM Response: {result}")
            return result
            
        except Exception as e:
            print(f"❌ [{self.name}] LLM call failed: {str(e)}")
            raise

class AnalystAgent(CypherAgent):
    """Agent responsible for analyzing questions and expanding context"""
    
    def __init__(self):
        super().__init__("Analyst", "Question Analysis & Context Expansion")
    
    def analyze_question(self, question: str, schema: str) -> Dict[str, str]:
        """Analyze the question and provide detailed context"""
        
        system_message = (
            "You are a data analysis expert specializing in graph databases and cybersecurity data. "
            "Your role is to analyze user questions and provide comprehensive context that will help "
            "other agents generate accurate Cypher queries."
        )
        
        prompt = f"""
        Analyze this question and provide detailed insights:
        
        Question: {question}
        Schema: {schema}
        
        Please provide:
        1. INTENT: What is the user trying to find out?
        2. KEY_ENTITIES: What nodes/relationships are involved?
        3. FILTERS: What conditions need to be applied?
        4. OUTPUT_FORMAT: How should results be presented?
        5. COMPLEXITY_NOTES: Any special considerations?
        6. SUGGESTED_APPROACH: Recommended query structure
        7. Respect given values, do not change them, values may take more than 1 word, use the complete given value
        
        Format your response as JSON with these keys.
        """
        
        response = self.call_llm(prompt, system_message)
        self.add_message(f"Analyzed question: {question}", {"analysis": response})
        
        try:
            # Try to parse as JSON, fallback to text parsing
            analysis = json.loads(response)
        except:
            # Fallback parsing if JSON fails
            analysis = {
                "INTENT": "Find information based on user question",
                "KEY_ENTITIES": "Entities mentioned in question",
                "FILTERS": "Conditions from question",
                "OUTPUT_FORMAT": "Count or list format",
                "COMPLEXITY_NOTES": "Standard query",
                "SUGGESTED_APPROACH": response
            }
        
        return analysis

class QueryGeneratorAgent(CypherAgent):
    """Agent responsible for generating Cypher queries"""
    
    def __init__(self):
        super().__init__("Generator", "Cypher Query Generation")
    
    def generate_query(self, question: str, schema: str, analysis: Dict[str, str]) -> str:
        """Generate a Cypher query based on analysis"""
        
        system_message = (
            "You are a Cypher query expert. Generate ONLY valid Cypher queries without explanations. "
            "You receive detailed analysis from another agent to help create accurate queries. "
            "CRITICAL: Use proper Cypher syntax - no nested MATCH in WHERE clauses. "
            "Use separate MATCH clauses or WITH clauses for complex queries."
        )
        
        prompt = f"""
        Generate a Cypher query based on this information:
        
        Original Question: {question}
        Schema: {schema}
        
        Analysis from Analyst Agent:
        Intent: {analysis.get('INTENT', 'Not specified')}
        Key Entities: {analysis.get('KEY_ENTITIES', 'Not specified')}
        Filters: {analysis.get('FILTERS', 'Not specified')}
        Output Format: {analysis.get('OUTPUT_FORMAT', 'Not specified')}
        Suggested Approach: {analysis.get('SUGGESTED_APPROACH', 'Not specified')}
        
        CRITICAL REQUIREMENTS:
        - Return ONLY the Cypher query - no explanations
        - Use read-only operations (MATCH, RETURN, WHERE, ORDER BY, LIMIT, WITH)
        - Properties like 'technique', 'protocol' are on relationships [:ATTACKS {{technique: "value"}}]
        - Attack relationships: (ip:IP)-[:ATTACKS]->(country:Country)
        - Use proper aggregation syntax
        - NO nested MATCH statements in WHERE clauses
        - Use WITH clauses or multiple MATCH clauses for complex queries
        - For subqueries, use WITH to pass results between MATCH clauses
        - Use complete given values, they may take more than 1 word

        

        Cypher Query:
        """
        
        query = self.call_llm(prompt, system_message)
        cleaned_query = self._clean_query(query)
        self.add_message(f"Generated query for: {question}", {"query": cleaned_query})
        
        return cleaned_query
    
    def _clean_query(self, query: str) -> str:
        """Clean and format the generated query"""
        # Remove markdown formatting
        query = re.sub(r'^```(?:cypher)?\n', '', query, flags=re.I | re.M)
        query = re.sub(r'\n```$', '', query, flags=re.M)
        query = query.strip('`\n ')
        
        # Remove explanations
        query = query.partition("**")[0]  # Remove any bold explanations
        query = query.partition("\n\n")[0]  # Take first paragraph only
        
        # Clean up
        query = query.strip().strip('"\'').rstrip(';')
        query = re.sub(r"^cypher\s*[:.]?\s*", "", query, flags=re.I)
        
        return query.strip()

class ValidatorAgent(CypherAgent):
    """Agent responsible for validating and correcting queries"""
    
    def __init__(self):
        super().__init__("Validator", "Query Validation & Correction")
    
    def validate_and_correct(self, query: str, question: str, schema: str, 
                           analysis: Dict[str, str], neo4j_driver, 
                           previous_attempts: List[QueryAttempt] = None) -> QueryAttempt:
        """Validate query and correct if needed"""
        
        print(f"🔍 [Validator] Validating query: {query}")
        
        # First check basic safety
        if not self._is_safe_query(query):
            return QueryAttempt(
                query=query,
                success=False,
                error="Query contains unsafe operations",
                agent=self.name
            )
        
        # Try to execute the query
        result = self._execute_query(query, neo4j_driver)
        
        if result.success:
            return result
        
        # If query failed, try to fix it
        print(f"🔧 [Validator] Query failed, attempting correction...")
        corrected_query = self._correct_query(
            query, question, schema, analysis, result.error, previous_attempts
        )
        
        if corrected_query != query:
            print(f"🔄 [Validator] Generated corrected query: {corrected_query}")
            return self._execute_query(corrected_query, neo4j_driver)
        else:
            return result
    
    def _is_safe_query(self, query: str) -> bool:
        """Check if query is safe to execute"""
        if WRITE_KEYWORDS.search(query):
            return False
        if ";" in query:
            return False
        return True
    
    def _execute_query(self, query: str, neo4j_driver) -> QueryAttempt:
        """Execute query and return result"""
        try:
            with neo4j_driver.session(database=NEO4J_DB) as s:
                # Validate syntax first
                s.run("EXPLAIN " + query).single()
                
                # Add safety limit
                safe_query = query
                if "LIMIT" not in query.upper():
                    safe_query = query + "\nLIMIT 500"
                
                # Execute
                rows = s.execute_read(lambda tx: tx.run(safe_query).data())
                
                return QueryAttempt(
                    query=query,
                    success=True,
                    results=rows,
                    agent=self.name
                )
                
        except Exception as e:
            return QueryAttempt(
                query=query,
                success=False,
                error=str(e),
                agent=self.name
            )
    
    def _correct_query(self, failed_query: str, question: str, schema: str, 
                      analysis: Dict[str, str], error: str, 
                      previous_attempts: List[QueryAttempt] = None) -> str:
        """Generate corrected query based on error"""
        
        system_message = (
            "You are a Cypher debugging expert. Your job is to fix broken Cypher queries. "
            "Return ONLY the corrected Cypher query with no explanations. "
            "CRITICAL: Never use nested MATCH statements in WHERE clauses - this is invalid Cypher syntax."
        )
        
        previous_errors = ""
        if previous_attempts:
            previous_errors = "\n\nPrevious failed attempts:\n"
            for i, attempt in enumerate(previous_attempts, 1):
                previous_errors += f"{i}. Query: {attempt.query}\n   Error: {attempt.error}\n"
        
        prompt = f"""
        Fix this broken Cypher query:
        
        Original Question: {question}
        Schema: {schema}
        Analysis Context: {analysis.get('SUGGESTED_APPROACH', 'No analysis available')}
        
        Failed Query: {failed_query}
        Error: {error}
        {previous_errors}
        
        CRITICAL FIXES NEEDED:
        - NEVER use nested MATCH in WHERE clauses like: WHERE ip.address IN (MATCH ...)
        - Use WITH clauses to pass data between MATCH statements
        - Properties on relationships: [:ATTACKS {{technique: "value"}}]
        - Correct relationship directions: (ip:IP)-[:ATTACKS]->(country:Country)
        - Proper WHERE syntax: WHERE r.protocol IN ["ssh", "http"]
        - Correct aggregation: RETURN r.technique, count(*) (not GROUP BY)
        
        
        Return ONLY the corrected Cypher query:
        """
        
        corrected = self.call_llm(prompt, system_message)
        corrected = self._clean_corrected_query(corrected)
        self.add_message(f"Corrected query with error: {error}", {"corrected_query": corrected})
        
        return corrected
    
    def _clean_corrected_query(self, query: str) -> str:
        """Clean the corrected query"""
        query = re.sub(r'^```(?:cypher)?\n', '', query, flags=re.I | re.M)
        query = re.sub(r'\n```$', '', query, flags=re.M)
        query = query.strip('`\n ').rstrip(';')
        return query.strip()

class MultiAgentCypherSystem:
    """Orchestrates the multi-agent system"""
    
    def __init__(self, neo4j_driver):
        self.neo4j_driver = neo4j_driver
        self.analyst = AnalystAgent()
        self.generator = QueryGeneratorAgent()
        self.validator = ValidatorAgent()
        self.conversation_log: List[AgentMessage] = []
        
    def log_message(self, agent: str, content: str, metadata: Optional[Dict] = None):
        """Log message to system conversation"""
        message = AgentMessage(agent, content, time.time(), metadata)
        self.conversation_log.append(message)
        
    def process_question(self, question: str, schema: str, max_retries: int = 3) -> Dict:
        """Process a question through the multi-agent system"""
        
        print(f"\n{'='*70}")
        print(f"🚀 MULTI-AGENT CYPHER SYSTEM STARTING")
        print(f"Question: {question}")
        print(f"{'='*70}")
        
        self.log_message("System", f"Processing question: {question}")
        
        # STEP 1: Analyst analyzes the question
        print(f"\n📊 STEP 1: ANALYST PHASE")
        print(f"{'─'*40}")
        analysis = self.analyst.analyze_question(question, schema)
        print(f"✅ [Analyst] Analysis complete")
        
        # STEP 2: Generator creates initial query
        print(f"\n🏗️ STEP 2: QUERY GENERATION PHASE")
        print(f"{'─'*40}")
        initial_query = self.generator.generate_query(question, schema, analysis)
        print(f"✅ [Generator] Initial query generated: {initial_query}")
        
        # STEP 3: Validator validates and potentially corrects
        print(f"\n🔍 STEP 3: VALIDATION & CORRECTION PHASE")
        print(f"{'─'*40}")
        
        attempts = []
        for attempt_num in range(1, max_retries + 2):  # +1 for initial attempt
            print(f"\n🔄 Attempt {attempt_num}/{max_retries + 1}")
            
            query_to_test = initial_query if attempt_num == 1 else attempts[-1].query
            result = self.validator.validate_and_correct(
                query_to_test, question, schema, analysis, 
                self.neo4j_driver, attempts
            )
            
            attempts.append(result)
            
            if result.success:
                print(f"🎉 SUCCESS on attempt {attempt_num}!")
                break
            else:
                print(f"❌ Attempt {attempt_num} failed: {result.error}")
                if attempt_num <= max_retries:
                    print(f"🔄 Preparing retry {attempt_num + 1}...")
        
        # Prepare final response
        final_result = attempts[-1]
        
        response = {
            "success": final_result.success,
            "query": final_result.query,
            "results": final_result.results if final_result.success else None,
            "error": final_result.error if not final_result.success else None,
            "attempts": len(attempts),
            "analysis": analysis,
            "conversation_log": [
                {
                    "agent": msg.agent,
                    "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self.conversation_log
            ]
        }
        
        print(f"\n{'='*70}")
        if final_result.success:
            print(f"✅ SYSTEM SUCCESS: Query executed successfully")
            print(f"📊 Results: {len(final_result.results)} rows returned")
        else:
            print(f"❌ SYSTEM FAILURE: All attempts exhausted")
            print(f"🔍 Final error: {final_result.error}")
        print(f"🔄 Total attempts: {len(attempts)}")
        print(f"{'='*70}")
        
        return response

# Main execution function
def generate_and_run_multiagent(question: str, schema_str: str, neo4j_driver, max_retries: int = 2):
    """Main function using the multi-agent system"""
    system = MultiAgentCypherSystem(neo4j_driver)
    return system.process_question(question, schema_str, max_retries)

# Example usage
if __name__ == "__main__":
    print("🔗 Connecting to Neo4j...")
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    schema = (
        "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
        "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, documented_create_date, documented_modified_date}]->(country:Country)\n"
        "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
    )
    
    questions = [
        "I need to know, from the ips that have used: 'T1001 Data Obfuscation' in the past, wich other unique techniques do they use in other attacks? sort them by the most used,  show count.",
        #"what is the most used technique? ","what are the most attacked industries?",
        #"Wich ASNs are using 'T1566 Phishing' against 'Government'? attacking wich countries? I need the top 3 ASNs that have attacked the most, and the countries they have targeted "
        #"I need to know if there are ASNs that perform on average 20% more attacks using 'T1566 Phishing' than the rest of asns (phishing specialized asns), and the percentaje of the attacks from that asn that are phishing"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🎯 PROCESSING QUESTION {i}")
        try:
            result = generate_and_run_multiagent(question, schema, drv)
            
            if result["success"]:
                print(f"\n📋 FINAL RESULTS:")
                results = result["results"]
                for j, row in enumerate(results[:10], 1):  # Show first 10
                    print(f"  {j}: {row}")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more rows")
            else:
                print(f"\n💥 FINAL ERROR: {result['error']}")
                
        except Exception as e:
            print(f"💥 System Error: {e}")
        
        if i < len(questions):
            print("\n" + "="*100)
    
    print("\n🔌 Closing Neo4j connection...")
    drv.close()
    print("✨ Multi-Agent System Demo Complete!")
