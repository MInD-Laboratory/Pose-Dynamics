def parse_participant_trial(filename: str) -> Tuple[str, int]:
    """Parse participant ID and trial number from pose filename.

    Args:
        filename: Pose filename (e.g., '3101_02_pose.csv')

    Returns:
        Tuple of (participant_id, trial_number)

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    base = Path(filename).name
    # Match pattern: PPPP_TT where PPPP is participant ID and TT is trial number
    m = re.match(r"^(\d{4})_(\d{2}).*\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern 'PPPP_TT_*.csv': {base}")
    return m.group(1), int(m.group(2))



def create_condition_mapping(participant_info: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """Create a mapping from participant ID and trial number to condition.

    Args:
        participant_info: DataFrame from load_participant_info()

    Returns:
        Nested dict: {participant_id: {trial_number: condition}}
        Example: mapping['3101'][1] returns 'L'
    """
    condition_map = {}

    for _, row in participant_info.iterrows():
        participant_id = str(row['Participant ID'])

        # Create trial mapping for this participant
        trial_map = {}

        # Parse each session (Session01, Session02, Session03)
        for session_num in [1, 2, 3]:
            session_col = f'session0{session_num}'
            if session_col in row and not pd.isna(row[session_col]) and row[session_col] != '-':
                session_value = str(row[session_col])

                # Extract condition letter (L, M, or H) - first character
                if session_value:
                    condition = session_value[0]  # L, M, or H
                    # Map trial number (session number) to condition
                    trial_map[session_num] = condition

        condition_map[participant_id] = trial_map

    return condition_map


def get_condition_for_file(filename: str, condition_map: Dict[str, Dict[int, str]]) -> str:
    """Get the condition (L/M/H) for a given pose file.

    Args:
        filename: Pose filename
        condition_map: Mapping from create_condition_mapping()

    Returns:
        Condition letter ('L', 'M', or 'H')

    Raises:
        ValueError: If no condition mapping found
    """
    participant_id, trial_num = parse_participant_trial(filename)

    if participant_id not in condition_map:
        raise ValueError(f"No condition mapping for participant {participant_id}")

    if trial_num not in condition_map[participant_id]:
        raise ValueError(f"No condition mapping for participant {participant_id} trial {trial_num}")

    return condition_map[participant_id][trial_num]
