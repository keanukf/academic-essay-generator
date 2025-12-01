"""CLI entry point for academic essay generator."""

import typer
from pathlib import Path
import yaml
from typing import Optional
from src.state.state import EssayState
from src.utils.ollama_client import OllamaClient
from src.loaders.pdf_loader import load_pdfs_from_directory
from src.graph.workflow import create_workflow
from src.utils.checkpoint import save_checkpoint, save_intermediate_essay

app = typer.Typer(help="Academic Essay Generator - Multi-agent system for generating PhD-level essays")


@app.callback(invoke_without_command=True)
def main(
    topic: str = typer.Option(..., "--topic", "-t", help="Essay topic"),
    criteria: str = typer.Option(..., "--criteria", "-c", help="Path to criteria.txt file"),
    literature: str = typer.Option(..., "--literature", "-l", help="Path to directory with PDFs"),
    output: str = typer.Option(..., "--output", "-o", help="Output path for essay.md"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml (default: ./config.yaml)"),
):
    """
    Generate an academic essay using the multi-agent pipeline.
    
    Example:
        python main.py --topic "The impact of transformer architectures on NLP" \\
            --criteria "criteria.txt" --literature ./inputs/ --output ./outputs/essay.md
    """
    # Load configuration
    config_path = Path(config) if config else Path("config.yaml")
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Initialize Ollama client
    ollama_config = config_data.get("ollama", {})
    ollama_client = OllamaClient(
        model=ollama_config.get("model", "llama3.1:8b-instruct-q4_K_M"),
        base_url=ollama_config.get("base_url", "http://localhost:11434"),
        timeout=ollama_config.get("timeout", 300)
    )
    
    # Load criteria
    criteria_path = Path(criteria)
    if not criteria_path.exists():
        typer.echo(f"Error: Criteria file not found: {criteria_path}", err=True)
        raise typer.Exit(1)
    
    with open(criteria_path, "r", encoding="utf-8") as f:
        criteria_text = f.read()
    
    # Load literature PDFs
    literature_path = Path(literature)
    if not literature_path.exists():
        typer.echo(f"Error: Literature directory not found: {literature_path}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"\n📚 Loading PDFs from {literature_path}...")
    chunking_config = config_data.get("chunking", {})
    literature_chunks = load_pdfs_from_directory(
        literature_path,
        chunk_size=chunking_config.get("chunk_size", 1000),
        chunk_overlap=chunking_config.get("chunk_overlap", 100)
    )
    
    if not literature_chunks:
        typer.echo("Warning: No literature chunks loaded. Continuing with empty literature.", err=True)
    
    typer.echo(f"✓ Loaded {len(literature_chunks)} literature chunks\n")
    
    # Initialize state
    initial_state = EssayState(
        topic=topic,
        criteria=criteria_text,
        literature_chunks=literature_chunks
    )
    
    # Create workflow
    review_config = config_data.get("review", {})
    workflow = create_workflow(
        ollama_client=ollama_client,
        review_threshold=review_config.get("threshold", 0.7),
        max_revision_cycles=review_config.get("max_revision_cycles", 2)
    )
    
    # Set up checkpoint directory (next to output file)
    output_path = Path(output)
    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"
    
    # Run workflow
    typer.echo("🚀 Starting essay generation pipeline...\n")
    typer.echo(f"💾 Checkpoints will be saved to: {checkpoint_dir}\n")
    
    # Initialize for error handling
    final_state_dict = None
    last_state_dict = None
    
    try:
        # Use streaming to capture intermediate states for checkpointing
        last_node = None
        
        # Track which nodes we've checkpointed to avoid duplicates
        checkpointed_nodes = set()
        
        # Stream workflow execution to capture intermediate states
        for event in workflow.stream(initial_state):
            # event is a dict with node names as keys
            for node_name, state_dict in event.items():
                last_node = node_name
                
                # Convert dict to EssayState for checkpointing
                current_state = EssayState(**state_dict)
                
                # Save checkpoints after key nodes
                if node_name in ["outline", "writer", "citation", "review"]:
                    # Only checkpoint once per revision cycle for each node
                    checkpoint_key = f"{node_name}_rev{current_state.revision_count}"
                    if checkpoint_key not in checkpointed_nodes:
                        try:
                            # Save full state checkpoint
                            checkpoint_file = save_checkpoint(current_state, checkpoint_dir, node_name)
                            
                            # Save intermediate essay if sections are available
                            if node_name in ["writer", "citation", "review"] and current_state.sections:
                                essay_file = save_intermediate_essay(current_state, checkpoint_dir, node_name)
                                if essay_file:
                                    typer.echo(f"  💾 Saved intermediate essay: {essay_file.name}")
                            
                            checkpointed_nodes.add(checkpoint_key)
                        except Exception as e:
                            typer.echo(f"  ⚠️  Warning: Failed to save checkpoint: {str(e)}", err=True)
                
                # Store state for checkpointing on error
                final_state_dict = state_dict
                last_state_dict = state_dict
        
        # If streaming didn't work or returned nothing, fall back to invoke
        if final_state_dict is None:
            typer.echo("  ⚠️  Streaming completed, using invoke as fallback...")
            final_state_dict = workflow.invoke(initial_state)
        
        if final_state_dict is None:
            typer.echo("Error: Workflow completed but no final state returned", err=True)
            raise typer.Exit(1)
        
        # Convert dict to EssayState instance
        final_state = EssayState(**final_state_dict)
        
        # Save final output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_state.final_essay)
        
        word_count = len(final_state.final_essay.split())
        typer.echo(f"\n✅ Essay generated successfully!")
        typer.echo(f"   Output: {output_path}")
        typer.echo(f"   Word count: {word_count}")
        typer.echo(f"   Revision cycles: {final_state.revision_count}")
        typer.echo(f"   Final review score: {final_state.review_score:.2f}/1.0")
        
    except KeyboardInterrupt:
        typer.echo("\n\n⚠️  Generation interrupted by user", err=True)
        # Try to save last checkpoint if available
        if last_state_dict:
            try:
                last_state = EssayState(**last_state_dict)
                save_checkpoint(last_state, checkpoint_dir, "interrupted")
                save_intermediate_essay(last_state, checkpoint_dir, "interrupted")
                typer.echo(f"💾 Last checkpoint saved to: {checkpoint_dir}")
            except Exception as save_error:
                typer.echo(f"  ⚠️  Could not save checkpoint: {str(save_error)}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\n❌ Error during generation: {str(e)}", err=True)
        # Try to save last checkpoint if available
        if last_state_dict:
            try:
                last_state = EssayState(**last_state_dict)
                save_checkpoint(last_state, checkpoint_dir, "error")
                save_intermediate_essay(last_state, checkpoint_dir, "error")
                typer.echo(f"💾 Last checkpoint saved to: {checkpoint_dir}")
            except Exception as save_error:
                typer.echo(f"  ⚠️  Could not save checkpoint: {str(save_error)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

