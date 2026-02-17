"""Tests for the interactive visualization module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.text import Text

from graphix.command import E, M, N, X, Z
from graphix.measurements import Measurement
from graphix.pattern import Pattern
from graphix.visualization_interactive import InteractiveGraphVisualizer


class TestInteractiveGraphVisualizer:
    @pytest.fixture
    def pattern(self) -> Pattern:
        """Fixture to provide a standard pattern for testing."""
        pattern = Pattern(input_nodes=[0, 1])
        pattern.add(N(node=0))
        pattern.add(N(node=1))
        pattern.add(N(node=2))
        pattern.add(E(nodes=(0, 1)))
        pattern.add(E(nodes=(1, 2)))
        pattern.add(M(node=0, measurement=Measurement.XY(0.5), s_domain={1}, t_domain={2}))
        pattern.add(M(node=1, measurement=Measurement.XY(0.0), s_domain={2}, t_domain=set()))
        pattern.add(X(node=2, domain={0}))
        pattern.add(Z(node=2, domain={1}))
        return pattern

    def test_init(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test initialization of the visualizer."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")

        # Capture the OpenGraph mock correctly
        mock_og_class = mocker.patch("graphix.visualization_interactive.OpenGraph")
        mock_og_instance = mock_og_class.return_value
        # Ensure infer_pauli_measurements returns a mock (or itself) to support chaining
        mock_og_instance.infer_pauli_measurements.return_value = mock_og_instance

        mocker.patch("matplotlib.pyplot.figure")

        # Mock layout generation
        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern)

        assert viz.total_steps == len(pattern)
        assert viz.enable_simulation
        # Check if get_layout was called
        mock_visualizer.assert_called_with(mock_og_instance)  # Verify visualizer init with corrected OG
        mock_vis_obj.get_layout.assert_called_once()
        # Check if node positions are set
        assert len(viz.node_positions) == 3

        # Check if infer_pauli_measurements was called
        mock_og_instance.infer_pauli_measurements.assert_called_once()

    def test_layout_generation(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test that layout logic delegates to GraphVisualizer."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        # Return specific positions to verify they are used
        expected_pos = {0: (10, 10), 1: (20, 20), 2: (30, 30)}
        mock_vis_obj.get_layout.return_value = (expected_pos, {}, {})

        viz = InteractiveGraphVisualizer(pattern)

        # Keys should match
        assert viz.node_positions.keys() == expected_pos.keys()
        # Values should be scaled by default node_distance (1, 1)
        assert viz.node_positions[0] == (10, 10)

    def test_update_graph_state_simulation_enabled(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test graph state update with simulation enabled."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")
        mock_backend = mocker.patch("graphix.visualization_interactive.StatevectorBackend")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        # Mock simulation backend
        backend_instance = mock_backend.return_value
        backend_instance.measure.return_value = 1

        viz = InteractiveGraphVisualizer(pattern, enable_simulation=True)

        # Update to the end of the pattern
        active, measured, _, _, results = viz._update_graph_state(len(pattern))

        # Basic Checks
        # Node 0 and 1 are measured
        assert 0 in measured
        assert 1 in measured
        # Node 2 is active
        assert 2 in active
        # Results should be populated (since we mocked measure return value)
        assert results[0] == 1
        assert results[1] == 1
        # Check if backend methods were called
        backend_instance.add_nodes.assert_called()
        backend_instance.entangle_nodes.assert_called()
        assert backend_instance.measure.call_count == 2
        assert backend_instance.correct_byproduct.call_count == 2

        # Manually trigger update to test drawing logic for measured/active nodes
        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.slider = MagicMock()
        viz.slider.val = len(pattern)
        viz._update(len(pattern))

        # Check that drawing methods were called
        # Measured nodes (0, 1) should generally be lightgray
        # Active node (2) should be white/blue
        assert viz.ax_graph.add_patch.call_count > 0
        assert viz.ax_graph.text.call_count > 0

    def test_update_graph_state_simulation_disabled(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test graph state update with simulation disabled."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern, enable_simulation=False)

        # Update to the end of the pattern
        active, measured, _, _, results = viz._update_graph_state(len(pattern))

        # Basic Checks
        # Node 0 and 1 are measured (topology tracking works without sim)
        assert 0 in measured
        assert 1 in measured
        # Node 2 is active
        assert 2 in active
        # Results should be empty as simulation is disabled
        assert results == {}

        # Manually trigger update to test drawing logic without simulation
        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.slider = MagicMock()
        viz.slider.val = len(pattern)
        viz._update(len(pattern))

        # Ensure text is drawn (commands, node labels)
        assert viz.ax_commands.text.call_count > 0
        assert viz.ax_graph.text.call_count > 0

    def test_navigation(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test step navigation methods."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern)
        # Mock slider
        viz.slider = MagicMock()
        viz.total_steps = 10
        viz.current_step = 5

        # Test Prev
        viz._prev_step(None)
        viz.slider.set_val.assert_called_with(4)

        # Test Next
        viz._next_step(None)
        viz.slider.set_val.assert_called_with(6)

        # Test Boundary Prev
        viz.current_step = 0
        viz.slider.reset_mock()
        viz._prev_step(None)
        viz.slider.set_val.assert_not_called()

        # Test Boundary Next
        viz.current_step = 10
        viz.slider.reset_mock()
        viz._next_step(None)
        viz.slider.set_val.assert_not_called()

    def test_visualize(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test the main visualize method (smoke test)."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")
        mock_show = mocker.patch("matplotlib.pyplot.show")
        mocker.patch("graphix.visualization_interactive.Slider")  # Mock Slider to avoid matplotlib validation

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.visualize()

        # Should show plot
        mock_show.assert_called_once()
        # Should initialize axes
        assert viz.ax_commands is not None
        assert viz.ax_graph is not None
        assert viz.slider is not None
        assert viz.btn_next is not None
        assert viz.btn_prev is not None

    def test_interaction_events(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test interaction event handlers."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.slider = MagicMock()

        # Test key press
        valid_key_event = MagicMock()
        valid_key_event.key = "right"
        # Mock return value of _next_step side effect
        mock_next = mocker.patch.object(viz, "_next_step")
        viz._on_key(valid_key_event)
        mock_next.assert_called_once()

        valid_key_event.key = "left"
        mock_prev = mocker.patch.object(viz, "_prev_step")
        viz._on_key(valid_key_event)
        mock_prev.assert_called_once()

    def test_z_correction_initialization(self, mocker: MagicMock) -> None:
        """Test tracking of Z corrections specifically to cover Z initialization."""
        # Create a pattern with a Z correction on a fresh node
        pattern = Pattern(input_nodes=[0])
        pattern.add(N(node=0))
        pattern.add(Z(node=0, domain=set()))

        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern, enable_simulation=False)

        # Trigger update to process the Z command
        viz.ax_graph = MagicMock()
        viz.ax_commands = MagicMock()
        viz.slider = MagicMock()
        viz._update(len(pattern))

        # Test pick event (clicking on command list)
        # We need a real Text object (or a mock that spec=Text) because _on_pick uses isinstance

        mock_artist = MagicMock(spec=Text)
        mock_artist.index = 5
        pick_event = MagicMock()
        pick_event.artist = mock_artist

        # Should set slider to index + 1 (highlight executed commands up to that point)
        viz._on_pick(pick_event)
        viz.slider.set_val.assert_called_with(6)

    def test_layout_fallback_triggered(self, mocker: MagicMock) -> None:
        """Test that spring layout fallback is triggered for narrow graphs."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")
        mock_spring_layout = mocker.patch("graphix.visualization_interactive.nx.spring_layout")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj

        initial_pos = {i: (0, i) for i in range(6)}
        mock_vis_obj.get_layout.return_value = (initial_pos, {}, {})

        # Spring layout returns something else
        spring_pos = {i: (10, i) for i in range(6)}
        mock_spring_layout.return_value = spring_pos

        big_pattern = Pattern(input_nodes=list(range(6)))
        for i in range(6):
            big_pattern.add(N(node=i))

        viz = InteractiveGraphVisualizer(big_pattern)

        # Check if spring layout was called
        mock_spring_layout.assert_called_once()

        # Check if positions were updated to spring layout positions
        # Note: InteractiveGraphVisualizer applies scaling after layout
        # default node_distance is (1, 1), so positions should match spring_pos
        assert viz.node_positions == spring_pos

    def test_draw_edges_coverage(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test that edge drawing logic is executed (covers lines 312-314)."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        # Provide positions for nodes 0, 1, 2
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.ax_graph = MagicMock()
        viz.slider = MagicMock()

        # Step 5 in our fixture pattern has entanglement E(0, 1) and E(1, 2)
        # and no nodes have been measured yet.
        viz._update(5)

        # Verify that plot (used for edges) was called
        # There should be 2 edges: (0, 1) and (1, 2)
        assert viz.ax_graph.plot.call_count == 2

    def test_draw_graph_exception_coverage(self, pattern: Pattern, mocker: MagicMock) -> None:
        """Test the exception handling in _draw_graph (covers lines 355-357)."""
        mock_visualizer = mocker.patch("graphix.visualization_interactive.GraphVisualizer")
        mocker.patch("graphix.visualization_interactive.OpenGraph")
        mocker.patch("matplotlib.pyplot.figure")

        mock_vis_obj = MagicMock()
        mock_visualizer.return_value = mock_vis_obj
        mock_vis_obj.get_layout.return_value = ({0: (0, 0), 1: (1, 0), 2: (0, 1)}, {}, {})

        viz = InteractiveGraphVisualizer(pattern)
        viz.ax_graph = MagicMock()

        # Force an exception during state update
        mocker.patch.object(viz, "_update_graph_state", side_effect=ValueError("Test Exception"))
        # Mock traceback to avoid cluttering test output
        mocker.patch("traceback.print_exc")

        # This should not raise but log/print
        viz._draw_graph()
