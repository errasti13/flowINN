class Checker:
    def _is_domain_closed(mesh: 'Mesh') -> bool:
        """
        Checks if all boundaries together form a single closed domain.
        Returns True if every vertex appears exactly twice and all boundaries are connected.
        """
        # Collect all segments from all boundaries
        segments = []
        for boundary_data in mesh.boundaries.values():
            x_coords = boundary_data['x']
            y_coords = boundary_data['y']
            
            for i in range(len(x_coords) - 1):
                pt1 = (float(x_coords[i]), float(y_coords[i]))
                pt2 = (float(x_coords[i + 1]), float(y_coords[i + 1]))
                segments.append((pt1, pt2))
        
        # Count vertex appearances in the entire domain
        endpoints = {}
        for seg in segments:
            for pt in seg:
                endpoints[pt] = endpoints.get(pt, 0) + 1
        
        # Check if each vertex appears exactly twice
        for pt, count in endpoints.items():
            if count != 2:
                print(f"Debug: Vertex {pt} appears {count} times (should be 2)")
                return False

        # Build adjacency graph
        adj = {}
        for seg in segments:
            p1, p2 = seg
            adj.setdefault(p1, set()).add(p2)
            adj.setdefault(p2, set()).add(p1)
        
        # DFS to check connectivity
        visited = set()
        def dfs(pt):
            visited.add(pt)
            for neighbor in adj[pt]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        # Start DFS from any vertex
        start = next(iter(adj))
        dfs(start)
        
        # Check if all vertices are connected
        if len(visited) != len(adj):
            print(f"Debug: Domain is not fully connected. Found {len(visited)} vertices, expected {len(adj)}")
            return False
            
        return True

    def check_closed_curve(mesh: 'Mesh') -> bool:
        """
        Checks if all boundaries together form a closed domain.
        Returns True if the domain is properly closed, False otherwise.
        """
        try:
            if not mesh.boundaries:
                print("Debug: No boundaries defined")
                return False
                
            if Checker._is_domain_closed(mesh):
                return True
            else:
                print("Debug: Domain is not properly closed")
                return False

        except Exception as e:
            print(f"Debug: Error checking closed curve: {str(e)}")
            return False