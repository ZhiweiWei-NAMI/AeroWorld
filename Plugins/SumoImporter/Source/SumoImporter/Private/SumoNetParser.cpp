#include "SumoNetParser.h"

#include "SumoImporterLog.h"
#include "XmlFile.h"
#include "XmlNode.h"
#include "Misc/DefaultValueHelper.h"

namespace
{
bool TryParseDouble(const FString& Value, double& OutValue)
{
	return FDefaultValueHelper::ParseDouble(Value, OutValue);
}

bool TryParseFloat(const FString& Value, float& OutValue)
{
	double Parsed = 0.0;
	if (!TryParseDouble(Value, Parsed))
	{
		return false;
	}

	OutValue = static_cast<float>(Parsed);
	return true;
}

bool TryParseInt(const FString& Value, int32& OutValue)
{
	return FDefaultValueHelper::ParseInt(Value, OutValue);
}

bool IsInternalEdge(const FString& FunctionValue)
{
	return FunctionValue.Equals(TEXT("internal"), ESearchCase::IgnoreCase);
}
}

bool FSumoNetParser::ParseShapePoints(const FString& ShapeString, TArray<FVector>& OutPointsM)
{
	OutPointsM.Reset();

	TArray<FString> PointTokens;
	ShapeString.ParseIntoArray(PointTokens, TEXT(" "), true);

	for (const FString& Token : PointTokens)
	{
		TArray<FString> Components;
		Token.ParseIntoArray(Components, TEXT(","), true);
		if (Components.Num() < 2)
		{
			continue;
		}

		double X = 0.0;
		double Y = 0.0;
		double Z = 0.0;
		if (!TryParseDouble(Components[0], X) || !TryParseDouble(Components[1], Y))
		{
			continue;
		}

		if (Components.Num() >= 3)
		{
			TryParseDouble(Components[2], Z);
		}

		OutPointsM.Add(FVector(X, Y, Z));
	}

	return OutPointsM.Num() > 0;
}

bool FSumoNetParser::ParseFile(
	const FString& FilePath,
	const FSumoParseOptions& ParseOptions,
	FSumoNetData& OutNetData,
	FSumoImportStats& OutStats,
	FString& OutError)
{
	OutNetData = FSumoNetData();
	OutStats = FSumoImportStats();
	OutError.Reset();

	FXmlFile XmlFile(FilePath, EConstructMethod::ConstructFromFile);
	if (!XmlFile.IsValid())
	{
		OutError = FString::Printf(TEXT("Invalid XML file: %s"), *FilePath);
		if (!XmlFile.GetLastError().IsEmpty())
		{
			OutError += FString::Printf(TEXT(" (%s)"), *XmlFile.GetLastError());
		}
		return false;
	}

	const FXmlNode* RootNode = XmlFile.GetRootNode();
	if (RootNode == nullptr || RootNode->GetTag() != TEXT("net"))
	{
		OutError = TEXT("Root node must be <net>.");
		return false;
	}

	for (const FXmlNode* ChildNode : RootNode->GetChildrenNodes())
	{
		if (ChildNode == nullptr)
		{
			continue;
		}

		const FString& Tag = ChildNode->GetTag();
		if (Tag == TEXT("location"))
		{
			OutNetData.Location.NetOffset = ChildNode->GetAttribute(TEXT("netOffset"));
			OutNetData.Location.ConvBoundary = ChildNode->GetAttribute(TEXT("convBoundary"));
			OutNetData.Location.OrigBoundary = ChildNode->GetAttribute(TEXT("origBoundary"));
			OutNetData.Location.ProjParameter = ChildNode->GetAttribute(TEXT("projParameter"));
		}
		else if (Tag == TEXT("edge"))
		{
			++OutStats.TotalEdges;

			FSumoEdgeData EdgeData;
			EdgeData.EdgeId = ChildNode->GetAttribute(TEXT("id"));
			EdgeData.FromJunction = ChildNode->GetAttribute(TEXT("from"));
			EdgeData.ToJunction = ChildNode->GetAttribute(TEXT("to"));
			EdgeData.Function = ChildNode->GetAttribute(TEXT("function"));
			EdgeData.EdgeType = ChildNode->GetAttribute(TEXT("type"));
			if (EdgeData.Function.IsEmpty())
			{
				EdgeData.Function = TEXT("normal");
			}

			if (!ParseOptions.bImportInternalEdges && IsInternalEdge(EdgeData.Function))
			{
				++OutStats.SkippedEdges;
				continue;
			}

			for (const FXmlNode* LaneNode : ChildNode->GetChildrenNodes())
			{
				if (LaneNode == nullptr || LaneNode->GetTag() != TEXT("lane"))
				{
					continue;
				}

				FSumoLaneData LaneData;
				LaneData.EdgeId = EdgeData.EdgeId;
				LaneData.LaneId = LaneNode->GetAttribute(TEXT("id"));

				const FString LaneIndexString = LaneNode->GetAttribute(TEXT("index"));
				if (!LaneIndexString.IsEmpty() && !TryParseInt(LaneIndexString, LaneData.LaneIndex))
				{
					OutStats.AddWarning(FString::Printf(TEXT("Invalid lane index for lane '%s'."), *LaneData.LaneId));
				}

				const FString SpeedString = LaneNode->GetAttribute(TEXT("speed"));
				if (!SpeedString.IsEmpty() && !TryParseFloat(SpeedString, LaneData.SpeedMps))
				{
					OutStats.AddWarning(FString::Printf(TEXT("Invalid lane speed for lane '%s'."), *LaneData.LaneId));
				}

				const FString LengthString = LaneNode->GetAttribute(TEXT("length"));
				if (!LengthString.IsEmpty() && !TryParseFloat(LengthString, LaneData.LengthM))
				{
					OutStats.AddWarning(FString::Printf(TEXT("Invalid lane length for lane '%s'."), *LaneData.LaneId));
				}

				const FString ShapeString = LaneNode->GetAttribute(TEXT("shape"));
				if (ShapeString.IsEmpty())
				{
					++OutStats.SkippedLanes;
					OutStats.AddWarning(FString::Printf(TEXT("Skip lane '%s': missing shape."), *LaneData.LaneId));
					continue;
				}

				if (!ParseShapePoints(ShapeString, LaneData.ShapePointsM) || LaneData.ShapePointsM.Num() < 2)
				{
					++OutStats.SkippedLanes;
					OutStats.AddWarning(FString::Printf(TEXT("Skip lane '%s': malformed shape."), *LaneData.LaneId));
					continue;
				}

				EdgeData.Lanes.Add(MoveTemp(LaneData));
			}

			if (EdgeData.Lanes.Num() == 0)
			{
				++OutStats.SkippedEdges;
				continue;
			}

			OutStats.ImportedEdges++;
			OutStats.ImportedLanes += EdgeData.Lanes.Num();
			OutNetData.Edges.Add(MoveTemp(EdgeData));
		}
		else if (Tag == TEXT("junction"))
		{
			FSumoJunctionData JunctionData;
			JunctionData.JunctionId = ChildNode->GetAttribute(TEXT("id"));

			const FString XString = ChildNode->GetAttribute(TEXT("x"));
			const FString YString = ChildNode->GetAttribute(TEXT("y"));
			const FString ZString = ChildNode->GetAttribute(TEXT("z"));

			TryParseDouble(XString, JunctionData.CenterM.X);
			TryParseDouble(YString, JunctionData.CenterM.Y);
			TryParseDouble(ZString, JunctionData.CenterM.Z);

			const FString ShapeString = ChildNode->GetAttribute(TEXT("shape"));
			ParseShapePoints(ShapeString, JunctionData.PolygonPointsM);

			OutNetData.Junctions.Add(MoveTemp(JunctionData));
		}
		else if (Tag == TEXT("connection"))
		{
			FSumoConnectionData ConnectionData;
			ConnectionData.FromEdge = ChildNode->GetAttribute(TEXT("from"));
			ConnectionData.ToEdge = ChildNode->GetAttribute(TEXT("to"));
			ConnectionData.ViaLane = ChildNode->GetAttribute(TEXT("via"));

			const FString FromLaneString = ChildNode->GetAttribute(TEXT("fromLane"));
			const FString ToLaneString = ChildNode->GetAttribute(TEXT("toLane"));
			if (!FromLaneString.IsEmpty())
			{
				TryParseInt(FromLaneString, ConnectionData.FromLane);
			}
			if (!ToLaneString.IsEmpty())
			{
				TryParseInt(ToLaneString, ConnectionData.ToLane);
			}

			OutNetData.Connections.Add(MoveTemp(ConnectionData));
		}
	}

	OutStats.JunctionCount = OutNetData.Junctions.Num();
	OutStats.ConnectionCount = OutNetData.Connections.Num();

	UE_LOG(
		LogSumoImporter,
		Log,
		TEXT("Parsed SUMO net: edges=%d importedEdges=%d importedLanes=%d junctions=%d connections=%d warnings=%d"),
		OutStats.TotalEdges,
		OutStats.ImportedEdges,
		OutStats.ImportedLanes,
		OutStats.JunctionCount,
		OutStats.ConnectionCount,
		OutStats.WarningCount);

	return true;
}
