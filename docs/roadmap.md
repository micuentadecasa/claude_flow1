# agentic creator Roadmap

generator of agentic solutions


## Development Workflow

1. **Task Planning**

- Study the existing codebase and understand the current state
- Update `ROADMAP.md` to include the new task
- Priority tasks should be inserted after the last completed task

2. **Task Creation**

- Study the existing codebase and understand the current state
- Create a new task file in the `/tasks` directory
- Name format: `XXX-description.md` (e.g., `001-db.md`)
- Include high-level specifications, relevant files, acceptance criteria, and implementation steps
- Refer to last completed task in the `/tasks` directory for examples. For example, if the current task is `012`, refer to `011` and `010` for examples.
- Note that these examples are completed tasks, so the content reflects the final state of completed tasks (checked boxes and summary of changes). For the new task, the document should contain empty boxes and no summary of changes. Refer to `000-sample.md` as the sample for initial state.

3. **Task Implementation**

- Follow the specifications in the task file
- Implement features and functionality
- Update step progress within the task file after each step
- Stop after completing each step and wait for further instructions

4. **Roadmap Updates**

- Mark completed tasks with ✅ in the roadmap
- Add reference to the task file (e.g., `See: /tasks/001-db.md`)

## Development Phases


- **Advanced Diff Algorithms**

  - Multiple diff granularity options (character, line, sentence level)
  - Configurable diff sensitivity settings
  - Whitespace and case-insensitive diff options
  - Performance optimization for large documents

- **Syntax-Aware Diff Highlighting**

  - Markdown syntax highlighting in diff view
  - Code block syntax highlighting within diffs
  - Preserve formatting and links in diff display
  - Rich text diff visualization

- **Interactive Diff Navigation**

  - Jump to next/previous change controls
  - Change filtering (additions, deletions, modifications)
  - Synchronized scrolling for side-by-side mode
  - Keyboard shortcuts for diff navigation
  - Change statistics and summary dashboard

- **Content Import/Scraping**

  - URL content extraction service
  - Metadata parsing
  - Multiple content type support

- **Advanced Prompt Templates**

  - Template variables and substitution
  - Prompt validation and versioning
  - Conditional logic in prompts
  - Prompt performance analytics

### Phase 4: Polish & Enhancement

- **Advanced Version Control**

  - Detailed version timeline
  - Advanced diff visualization
  - Conflict resolution

- **Advanced Search & Filtering**

  - Global search
  - Advanced filtering
  - Search result highlighting

- **Export Features**
  - Basic format exports (Markdown, HTML)
  - Platform-specific exports
  - Batch export

### Phase 5: Advanced Features

- **Advanced AI Editing Features**

  - Enhanced editing commands with more options (rewrite, formalize, casual, fix-grammar, add-examples, restructure)
  - Paragraph-level editing: users can select specific paragraphs to apply targeted edits
  - Advanced diff visualization before applying changes (side-by-side, interactive)
  - Separate system prompt management for editing commands (different from post creation)
  - Enhanced content extraction: better parsing of AI responses including markdown code blocks
  - Custom editing command creation and templates
  - Command template management and sharing system
  - Command execution history and analytics
  - Database-driven command management system

### Phase 6: Team & Collaboration

- **Archive System**

  - Archive/unarchive posts
  - Archived posts view
  - Permanent deletion

- **Team Collaboration**
  - Team member roles
  - Collaborative editing
  - Comment system