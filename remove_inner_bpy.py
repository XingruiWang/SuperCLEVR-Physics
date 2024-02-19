import bpy
import bmesh

# Get the active camera in the scene
camera = bpy.context.scene.camera

# Ensure there is an active camera
if camera is not None:
    # Set up ray cast
    ray_cast = bpy.context.scene.ray_cast

    # Create a list to store the selected vertices
    selected_vertices = []

    # Iterate through all mesh objects in the scene
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Ensure the object is visible in the viewport
            if obj.hide_viewport:
                continue

            # Create a BMesh object from the mesh
            bm = bmesh.new()
            bm.from_mesh(obj.data)

            # Transform mesh vertices to world space
            world_matrix = obj.matrix_world
            mesh_vertices = [world_matrix @ v.co for v in bm.verts]

            # Perform ray casting from the camera to each vertex
            for v in bm.verts:
                result, location, normal, face_index = obj.ray_cast(
                    camera.location, (v.co - camera.location).normalized())

                # Check if the vertex is visible
                if result:
                    selected_vertices.append(v)

            # Free the BMesh
            bm.free()

    # Deselect all vertices
    bpy.ops.mesh.select_all(action='DESELECT')

    # Select the visible vertices
    for v in selected_vertices:
        v.select = True

    # Update the mesh selection
    bpy.context.view_layer.objects.active = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

else:
    print("No active camera in the scene.")
