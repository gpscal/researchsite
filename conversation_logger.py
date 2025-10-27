#!/usr/bin/env python3
"""
Conversation Logger for Training Data Collection

Captures LLM conversations and formats them for fine-tuning custom models.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ConversationLogger:
    """
    Logs LLM conversations in a format suitable for training data.
    
    Format: JSONL with each line containing:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "metadata": {
            "timestamp": "...",
            "model": "...",
            "domain": "...",
            "session_id": "...",
            "rating": null  # Can be added later for quality filtering
        }
    }
    """
    
    def __init__(self, 
                 output_dir: str = "data/training_data/conversations",
                 min_conversation_length: int = 2,
                 auto_save: bool = True):
        """
        Initialize conversation logger.
        
        Args:
            output_dir: Directory to save conversation logs
            min_conversation_length: Minimum messages to save (default: 2 = 1 exchange)
            auto_save: Whether to auto-save after each conversation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_conversation_length = min_conversation_length
        self.auto_save = auto_save
        
        # Current session conversations (in memory)
        self.conversations = []
        
        logger.info(f"ConversationLogger initialized: {self.output_dir}")
    
    def log_conversation(self,
                        messages: List[Dict],
                        model: str = "unknown",
                        domain: str = "general",
                        session_id: Optional[str] = None,
                        rating: Optional[int] = None,
                        metadata: Optional[Dict] = None) -> bool:
        """
        Log a single conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name used (e.g., 'gpt-4o', 'gpt-5')
            domain: Domain/category (e.g., 'code_assistant', 'pcap_analysis', 'research')
            session_id: Optional session identifier
            rating: Optional quality rating (1-5)
            metadata: Additional metadata
        
        Returns:
            bool: True if logged successfully
        """
        # Filter out empty messages
        messages = [m for m in messages if m.get('content', '').strip()]
        
        # Only log if we have meaningful conversation
        if len(messages) < self.min_conversation_length:
            logger.debug(f"Conversation too short ({len(messages)} messages), skipping")
            return False
        
        # Create conversation entry
        conversation = {
            "messages": messages,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "domain": domain,
                "session_id": session_id or self._generate_session_id(),
                "rating": rating,
                "message_count": len(messages),
                **(metadata or {})
            }
        }
        
        self.conversations.append(conversation)
        
        if self.auto_save:
            self.save_conversations()
        
        logger.info(f"Logged conversation: {len(messages)} messages, domain={domain}")
        return True
    
    def log_exchange(self,
                     user_message: str,
                     assistant_message: str,
                     system_prompt: Optional[str] = None,
                     **kwargs) -> bool:
        """
        Convenience method to log a single user-assistant exchange.
        
        Args:
            user_message: User's input
            assistant_message: Assistant's response
            system_prompt: Optional system prompt
            **kwargs: Additional arguments for log_conversation
        
        Returns:
            bool: True if logged successfully
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ])
        
        return self.log_conversation(messages, **kwargs)
    
    def save_conversations(self, filename: Optional[str] = None) -> str:
        """
        Save conversations to JSONL file.
        
        Args:
            filename: Optional custom filename (defaults to timestamped)
        
        Returns:
            str: Path to saved file
        """
        if not self.conversations:
            logger.debug("No conversations to save")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations_{timestamp}.jsonl"
        
        output_path = self.output_dir / filename
        
        # Append to existing file if it exists, otherwise create new
        mode = 'a' if output_path.exists() else 'w'
        
        with open(output_path, mode, encoding='utf-8') as f:
            for conversation in self.conversations:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        count = len(self.conversations)
        self.conversations.clear()  # Clear after saving
        
        logger.info(f"Saved {count} conversations to {output_path}")
        return str(output_path)
    
    def export_for_training(self, 
                           input_file: Optional[str] = None,
                           output_file: Optional[str] = None,
                           filter_domain: Optional[str] = None,
                           min_rating: Optional[int] = None,
                           include_system_prompt: bool = True) -> Dict:
        """
        Export conversations in format ready for fine-tuning.
        
        Args:
            input_file: Input JSONL file (defaults to latest)
            output_file: Output file for training data
            filter_domain: Only include specific domain
            min_rating: Only include conversations with rating >= this
            include_system_prompt: Whether to include system prompts
        
        Returns:
            dict: Statistics about exported data
        """
        # Find input file
        if input_file is None:
            # Get most recent conversation file
            files = sorted(self.output_dir.glob("conversations_*.jsonl"))
            if not files:
                return {"success": False, "error": "No conversation files found"}
            input_file = files[-1]
        else:
            input_file = Path(input_file)
        
        if output_file is None:
            output_file = self.output_dir.parent / "training" / f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read and filter conversations
        exported_count = 0
        filtered_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as inf, \
             open(output_file, 'w', encoding='utf-8') as outf:
            
            for line in inf:
                conversation = json.loads(line.strip())
                
                # Apply filters
                metadata = conversation.get("metadata", {})
                
                if filter_domain and metadata.get("domain") != filter_domain:
                    filtered_count += 1
                    continue
                
                if min_rating and (metadata.get("rating") or 0) < min_rating:
                    filtered_count += 1
                    continue
                
                # Format for training
                messages = conversation["messages"]
                
                if not include_system_prompt:
                    messages = [m for m in messages if m["role"] != "system"]
                
                training_example = {
                    "messages": messages,
                    "domain": metadata.get("domain", "general")
                }
                
                outf.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                exported_count += 1
        
        stats = {
            "success": True,
            "input_file": str(input_file),
            "output_file": str(output_file),
            "exported": exported_count,
            "filtered": filtered_count,
            "total": exported_count + filtered_count
        }
        
        logger.info(f"Exported {exported_count} conversations to {output_file}")
        return stats
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def get_stats(self, input_file: Optional[str] = None) -> Dict:
        """
        Get statistics about logged conversations.
        
        Args:
            input_file: Optional specific file to analyze
        
        Returns:
            dict: Statistics
        """
        if input_file is None:
            files = list(self.output_dir.glob("conversations_*.jsonl"))
            if not files:
                return {"total_conversations": 0, "files": 0}
            input_file = files[-1]
        else:
            input_file = Path(input_file)
        
        if not input_file.exists():
            return {"error": "File not found"}
        
        stats = {
            "file": str(input_file),
            "total_conversations": 0,
            "by_domain": {},
            "by_model": {},
            "avg_messages": 0,
            "with_rating": 0
        }
        
        total_messages = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                conversation = json.loads(line.strip())
                stats["total_conversations"] += 1
                
                metadata = conversation.get("metadata", {})
                domain = metadata.get("domain", "unknown")
                model = metadata.get("model", "unknown")
                
                stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
                stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
                
                message_count = len(conversation.get("messages", []))
                total_messages += message_count
                
                if metadata.get("rating"):
                    stats["with_rating"] += 1
        
        if stats["total_conversations"] > 0:
            stats["avg_messages"] = total_messages / stats["total_conversations"]
        
        return stats


# Global instance for easy access
_logger_instance = None

def get_conversation_logger(output_dir: str = "data/training_data/conversations") -> ConversationLogger:
    """Get or create global conversation logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ConversationLogger(output_dir=output_dir)
    return _logger_instance

